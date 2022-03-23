# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Analysis used in TensorIR scheduling"""
from typing import List, Optional

from ..buffer import Buffer
from ..stmt import For
from ..expr import PrimExpr
from ..function import IndexMap

from . import _ffi_api


def suggest_index_map(
    buffer: Buffer,
    indices: List[PrimExpr],
    loops: List[For],
    predicate: PrimExpr,
) -> Optional[IndexMap]:
    """Provided the access pattern to a buffer, suggest one of the possible layout
    transformation to maximize the locality of the access pattern.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be transformed.
    indices : List[PrimExpr]
        The access pattern to the buffer.
    loops : List[For]
        The loops above the buffer.
    predicate : PrimExpr
        The predicate of the access.

    Returns
    -------
    index_map : Optional[IndexMap]
        The suggested index map. None if no transformation is suggested.
    """
    return _ffi_api.SuggestIndexMap(  # type: ignore # pylint: disable=no-member
        buffer,
        indices,
        loops,
        predicate,
    )
