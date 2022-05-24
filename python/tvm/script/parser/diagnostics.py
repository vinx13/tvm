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
# pylint: disable=missing-docstring
from tvm.ir import IRModule, SourceName, Span, diagnostics

from . import doc
from .source import Source


class Diagnostics:

    source: Source
    ctx: diagnostics.DiagnosticContext

    def __init__(self, source: Source):
        mod = IRModule()
        mod.source_map.add(source.source_name, source.full_source)
        self.source = source
        self.ctx = diagnostics.DiagnosticContext(mod, diagnostics.get_renderer())

    def _emit(self, node: doc.AST, message: str, level: diagnostics.DiagnosticLevel) -> None:
        lineno = node.lineno or self.source.start_line
        col_offset = node.col_offset or self.source.start_column
        end_lineno = node.end_lineno or lineno
        end_col_offset = node.end_col_offset or col_offset
        lineno += self.source.start_line - 1
        end_lineno += self.source.start_line - 1
        col_offset += self.source.start_column + 1
        end_col_offset += self.source.start_column + 1
        self.ctx.emit(
            diagnostics.Diagnostic(
                level=level,
                span=Span(
                    source_name=SourceName(self.source.source_name),
                    line=lineno,
                    end_line=end_lineno,
                    column=col_offset,
                    end_column=end_col_offset,
                ),
                message=message,
            )
        )

    def error(self, node: doc.AST, message: str) -> None:
        self._emit(node, message, diagnostics.DiagnosticLevel.ERROR)
        self.ctx.render()
