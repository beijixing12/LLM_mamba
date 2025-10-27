# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import csv
import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from elasticsearch import Elasticsearch

try:
    from elastic_transport import ConnectionError as ESConnectionError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency surface
    try:
        from elasticsearch import ConnectionError as ESConnectionError  # type: ignore
    except ImportError:  # pragma: no cover
        ESConnectionError = ()  # type: ignore


def _resolve_npz_key(
    available: Iterable[str],
    primary: str,
    alternatives: Sequence[str],
) -> Optional[str]:
    """Return the first matching key present in ``available``.

    Historic preprocessing pipelines occasionally wrote different column names
    into the compressed ``.npz`` archives (for example ``skill_seq`` instead of
    ``skill``).  The current exporters use the canonical keys, but keeping a
    compatibility shim here avoids runtime crashes when users plug in older
    artefacts.  The helper prefers ``primary`` and otherwise scans the supplied
    ``alternatives``.
    """

    available_set = list(available)
    if primary in available_set:
        return primary
    for candidate in alternatives:
        if candidate in available_set:
            return candidate
    return None


def _load_concept_metadata(
    data_folder: str,
    dataset_name: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load concept identifiers (and optionally names) from a mapping CSV.

    Older preprocessing pipelines for ASSIST-style datasets only wrote the
    ``knowledge_concept_mapping`` artefact, so the compressed ``.npz`` file may
    lack the ``concept_ids`` array required to derive the complete skill
    vocabulary size.  This helper scans a couple of conventional filenames and
    returns numpy arrays that mirror the structure exposed by the newer
    preprocessors.  The fallback keeps the runtime loader backwards compatible
    while still allowing downstream components to align with prerequisite
    graphs that expect the full concept set.
    """

    candidates = (
        os.path.join(data_folder, f"knowledge_concept_mapping_{dataset_name}.csv"),
        os.path.join(data_folder, "knowledge_concept_mapping.csv"),
    )

    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue

        concept_ids: list[int] = []
        concept_names: list[str] = []

        with open(candidate, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header_consumed = False
            for row in reader:
                if not header_consumed and row and row[0] == "concept_id":
                    header_consumed = True
                    continue
                if not row:
                    continue
                try:
                    concept_id = int(float(row[0]))
                except (TypeError, ValueError):
                    continue
                concept_ids.append(concept_id)
                if len(row) > 1:
                    concept_names.append(row[1])

        if concept_ids:
            ordered_ids = np.array(concept_ids, dtype=np.int64)
            ordered_names: Optional[np.ndarray]
            if concept_names and len(concept_names) == len(concept_ids):
                ordered_names = np.array(concept_names, dtype=object)
            else:
                ordered_names = None
            return ordered_ids, ordered_names

    return None, None


class KTDataset:
    def __init__(self, data_folder, max_len=200):
        folder_name = os.path.basename(data_folder)
        self.dataset_name = folder_name
        concept_ids = None
        concept_names = None
        npz_path = os.path.join(data_folder, folder_name + '.npz')
        with np.load(npz_path, allow_pickle=True) as data:
            files = data.files

            skill_key = _resolve_npz_key(
                files,
                'skill',
                (
                    'skills',
                    'skill_seq',
                    'skill_seqs',
                    'skill_sequence',
                    'concept_seq',
                    'concept_seqs',
                    'kc_seq',
                    'kc_seqs',
                ),
            )
            y_key = _resolve_npz_key(files, 'y', ('label', 'labels', 'correct'))
            len_key = _resolve_npz_key(files, 'real_len', ('seq_len', 'seq_lens', 'lengths'))

            missing_keys = [
                name
                for name, resolved in (
                    ('skill', skill_key),
                    ('y', y_key),
                    ('real_len', len_key),
                )
                if resolved is None
            ]
            if missing_keys:
                missing_list = ', '.join(missing_keys)
                available_keys = ', '.join(files)
                raise KeyError(
                    f"{missing_list} is not a file in the archive; available keys: {available_keys}; "
                    f"while reading {npz_path}"
                )

            resolved_keys = [skill_key, y_key, len_key]
            self.data = [data[key] for key in resolved_keys]
            if folder_name == 'junyi':
                self.data[0] = data['problem'] - 1
            if 'concept_ids' in data.files:
                concept_ids = np.asarray(data['concept_ids'], dtype=np.int64)
            if 'concept_names' in data.files:
                concept_names = np.asarray(data['concept_names'], dtype=object)

        if concept_ids is None:
            mapping_ids, mapping_names = _load_concept_metadata(data_folder, folder_name)
            concept_ids = mapping_ids if mapping_ids is not None else None
            if mapping_names is not None:
                concept_names = mapping_names

        self.data[1] = [_.astype(np.float32) for _ in self.data[1]]
        self.concept_ids = concept_ids
        self.concept_names = concept_names

        skills = self.data[0]
        max_from_sequences = -1
        try:
            max_from_sequences = int(np.max(skills))
        except (ValueError, TypeError):
            try:
                concatenated = np.concatenate(skills)
            except ValueError:
                concatenated = np.array([], dtype=np.int64)
            if concatenated.size:
                max_from_sequences = int(concatenated.max())

        max_from_mapping = -1
        if concept_ids is not None and concept_ids.size:
            max_from_mapping = int(concept_ids.max())
        mapping_length = -1
        if concept_ids is not None:
            mapping_length = int(concept_ids.size)

        max_skill = max(max_from_sequences, max_from_mapping)
        mapped_skill_num = mapping_length if mapping_length >= 0 else 0
        inferred = max_skill + 1 if max_skill >= 0 else 0
        self.feats_num = max(inferred, mapped_skill_num)
        self.data = list(zip(*self.data))
        self.users_num = len(self.data)
        self.max_len = max_len
        self.mask = np.zeros(self.max_len, dtype=bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        skill, y, real_len = self.data[item]
        skill, y = skill[:self.max_len], y[:self.max_len]
        if len(skill) < self.max_len:
            skill, y = skill.copy(), y.copy()
            skill.resize((self.max_len,))
            y.resize((self.max_len,))
        mask = self.mask.copy()
        mask[:real_len] = True
        return skill, y, mask


class RecDataset(KTDataset):
    # For GRU4Rec, Predict the next item
    def __getitem__(self, item):
        skill, _, real_len = self.data[item]
        skill = skill[:self.max_len + 1]
        if len(skill) < self.max_len + 1:
            skill = skill.copy()
            skill.resize((self.max_len + 1,))
        mask = self.mask.copy()
        mask[:real_len - 1] = True
        return skill[:-1], skill[1:], mask


class RetrievalDataset(KTDataset):
    def __init__(self, data_folder, r=5, train_test_split=0.8, max_len=200):
        super(RetrievalDataset, self).__init__(data_folder, max_len)
        self.es = Elasticsearch(hosts=['http://localhost:9200/']).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
        )
        self._es_available = True
        self.safe_users = np.arange(int(len(self.data) * train_test_split))
        self.R = r
        self.index = f'{self.dataset_name}_train'
        self.safe_query = self.get_safe_query()

    def get_safe_query(self):
        safe_query = [[self.data[i][0][0], self.data[i][1][0]] for i in self.safe_users]
        safe_query = np.asarray(safe_query, dtype=np.int32)
        return safe_query

    def _empty_retrieval_response(self):
        r_his = np.zeros((self.max_len * self.R, self.max_len, 2), dtype=np.int32)
        r_skill_y = np.zeros((self.max_len, self.R, 2), dtype=np.int32)
        r_len = np.ones((self.max_len, self.R), dtype=np.int32)
        return r_his, r_skill_y, r_len

    def get_query(self, user, skills, index_range):
        safe_user = np.random.choice(self.safe_users, self.R + 1, replace=False)
        safe_user = safe_user[safe_user != user][:self.R]
        safe_query = self.safe_query[safe_user]
        query_s = []
        skills_str = ' '
        query_indices = []
        max_index = len(skills) - 1
        for raw_idx in index_range:
            idx = int(raw_idx)
            if idx < 0:
                continue
            if idx > max_index:
                break
            query_indices.append(idx)

        if not query_indices:
            return self._empty_retrieval_response()

        for idx in query_indices:
            skills_str += f' {skills[idx]}'
            query = [
                {'index': self.index},
                {'size': self.R,
                 'query': {'bool': {'must': [{'term': {'skill': skills[idx]}},
                                              {'match': {'history': skills_str}}],
                                     'must_not': {'term': {'user': user}}}}}
            ]
            query_s += query
        if not self._es_available or self.es is None:
            return self._empty_retrieval_response()
        try:
            result = self.es.msearch(searches=query_s)['responses']
        except Exception as exc:  # pragma: no cover - network failure handling
            matched_es_error = isinstance(exc, ESConnectionError)
            if matched_es_error:
                self._es_available = False
                return self._empty_retrieval_response()
            raise
        r_his, r_skill_y, r_len = [], [], []
        for rs in result:  # seq_len
            skill_y, real_len = [], []
            rs = rs['hits']['hits']
            for r in rs:  # R
                r = r['_source']
                his = np.fromstring(r['history'], dtype=np.int32, sep=' ')
                his = np.stack((his, np.asarray(r['y'], dtype=np.int32)), axis=-1)
                if his.ndim == 1:
                    his = np.expand_dims(his, 0)
                his.resize((self.max_len, 2))
                r_his.append(his)
                skill_y.append(his[-1])
                real_len.append(len(skill_y))
            # If the quantity is less than R, fill it up
            for _ in range(self.R - len(rs)):
                his = safe_query[_:_ + 1].copy()
                his.resize((self.max_len, 2))
                r_his.append(his)
                skill_y.append(his[_])
                real_len.append(len(skill_y))
            r_skill_y.append(skill_y)  # (R, 2)
            r_len.append(real_len)
        r_his, r_skill_y, r_len = np.asarray(r_his, dtype=np.int32), np.asarray(r_skill_y, dtype=np.int32), np.asarray(
            r_len, dtype=np.int32)
        r_his.resize((self.max_len * self.R, self.max_len, 2))
        r_skill_y.resize((self.max_len, self.R, 2))
        r_len.resize((self.max_len, self.R))
        r_len[r_len < 1] = 1
        return r_his, r_skill_y, r_len

    def __getitem__(self, item):
        skill, y, real_len = self.data[item]
        skill, y = skill[:self.max_len], y[:self.max_len]
        try:
            real_len_int = int(real_len)
        except (TypeError, ValueError):
            real_len_int = 0
        if real_len_int < 0:
            real_len_int = 0
        effective_len = min(real_len_int, len(skill))
        if effective_len <= 0:
            r_his, r_skill_y, r_len = self._empty_retrieval_response()
        else:
            r_his, r_skill_y, r_len = self.get_query(item, skill, range(effective_len))
        if len(skill) < self.max_len:
            skill, y = skill.copy(), y.copy()
            skill.resize((self.max_len,))
            y.resize((self.max_len,))
        mask = self.mask.copy()
        mask[:effective_len] = True
        return skill, r_his, r_skill_y, y, mask, r_len


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    folder = '../data/assist15'
    dataset = RetrievalDataset(folder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for batch in tqdm(dataloader):
        pass
