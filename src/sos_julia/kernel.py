#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

from collections import Sequence
import tempfile
from sos.utils import short_repr, env
from IPython.core.error import UsageError
import pandas
import numpy


def homogeneous_type(seq):
    iseq = iter(seq)
    first_type = type(next(iseq))
    if first_type in (int, float):
        return True if all(isinstance(x, (int, float)) for x in iseq) else False
    else:
        return True if all(isinstance(x, first_type) for x in iseq) else False


'''
NaN		None
Bool		boolean
Int64		integer
Float64		double
Char		str
Complex{}		complex
Float64		numpy.float64
String		str
Array{,1}		Sequence (list,tuple, ...)
Set{}		set
Dict{}		dict
Array{Int64/Float64,2}	2 dimensions	numpy.matrix
Array{Int64/Float64,}	> 2 dimensions	numpy.ndarray
NamedArrays.NamedArray	1 dimension	pandas.Series
Dataframes.Dataframe		pandas.DataFrame
'''

julia_install_package = {
    'feather':
        r'''
try
  using Feather
catch
  Pkg.add("Feather")
  using Feather
end
''',
    'namedarray':
        r'''
try
  using NamedArrays
catch
  Pkg.add("NamedArrays")
  using NamedArrays
end
''',
    'dataframes':
        r'''
try
  using DataFrames
catch
  Pkg.add("DataFrames")
  using DataFrames
end
'''
}

julia_init_statements = r'''
function __julia_py_repr_logical_1(obj)
    obj==true ? "True" : "False"
end
function __julia_py_repr_integer_1(obj)
    return string(obj)
end
function __julia_py_repr_double_1(obj)
    return "numpy.float64(" * string(obj) * ")"
end
function __julia_py_repr_complex_1(obj)
  rl = real(obj)
  im = imag(obj)
  return "complex(" * string(rl) * "," * string(im) * ")"
end
function __julia_py_repr_character_1(obj)
  return "r\"\"\"" * obj * "\"\"\""
end
function __julia_py_repr_dict_1(obj)
  val = collect(values(obj))
  key = collect(keys(obj))
  res = __julia_py_repr_character_1(key[1]) * ":" * string(__julia_py_repr(val[1])) * ","
  for i in 2:length(val)
    res = res * __julia_py_repr_character_1(key[i]) * ":" * string(__julia_py_repr(val[i])) * ","
  end
  return "{" * res * "}"
end
# Dataframe in Julia doesn't have rowname. Will keep tracking any update of Dataframes package in Julia
function __julia_py_repr_dataframe(obj)
  tf = joinpath(tempname())
  if !isdefined(@__MODULE__, :Feather)
    return "SOS_JULIA_REQUIRE:feather"
  end
  Feather.write(tf, obj)
  return "read_dataframe(r'" * tf * "')"
end
function __julia_py_repr_matrix(obj)
  tf = joinpath(tempname())
  if !isdefined(@__MODULE__, :DataFrame)
    return "SOS_JULIA_REQUIRE:dataframes"
  end
  if !isdefined(@__MODULE__, :Feather)
    return "SOS_JULIA_REQUIRE:feather"
  end
  Feather.write(tf, convert(DataFrame, obj))
  return "numpy.asmatrix(read_dataframe(r'" * tf * "'))"
end
# namedarray is specific for list with names (and named vector in R)
function __julia_py_repr_namedarray(obj)
  key = names(obj)[1]
  val = [obj[i] for i in 1:length(key)]
  return "pandas.Series(" * "[" * join([__julia_py_repr(i) for i in val], ",") * "]," * "index=[" * join([__julia_py_repr(j) for j in key], ",") * "])"
end
function __julia_py_repr_set(obj)
  return "{" * join([__julia_py_repr(i) for i in obj], ",") * "}"
end
function __julia_py_repr_n(obj)
  # The problem of join() is that it would ignore the double quote of a string
  return "[" * join([__julia_py_repr(i) for i in obj], ",") * "]"
end
function __julia_has_row_names(df)
  return !(names(df)[1]==collect(1:size(df)[1]))
end
function __julia_has_col_names(df)
  return !(names(df)[2]==collect(1:size(df)[2]))
end
function __julia_py_repr(obj)
  if isa(obj, Matrix)
    __julia_py_repr_matrix(obj)
  elseif isa(obj, Set)
    __julia_py_repr_set(obj)
  # type of NaN in Julia is Float64
  elseif isa(obj, Cvoid) || obj === NaN
    return "None"
  elseif isa(obj, Dict)
    __julia_py_repr_dict_1(obj)
    # if needed to name vector in julia, need to use a package called NamedArrays
  elseif isa(obj, Vector{Int})
    if (length(obj) == 1)
      __julia_py_repr_integer_1(obj)
    else
      return "numpy.array([" * join([__julia_py_repr_integer_1(i) for i in obj], ",") * "])"
    end
  elseif isa(obj, Vector{Complex{Int}}) || isa(obj, Vector{Complex{Float64}})
    if (length(obj) == 1)
      __julia_py_repr_complex_1(obj)
    else
      return "[" * join([__julia_py_repr_complex_1(i) for i in obj], ",") * "]"
    end
  elseif isa(obj, Vector{Float64})
    if (length(obj) == 1)
      __julia_py_repr_double_1(obj)
    else
      return "numpy.array([" * join([__julia_py_repr_double_1(i) for i in obj], ",") * "], dtype=numpy.float64)"
    end
  elseif isa(obj, Vector{String})
    if (length(obj) == 1)
      __julia_py_repr_character_1(obj)
    else
      return "[" * join([__julia_py_repr_character_1(i) for i in obj], ",") * "]"
    end
  elseif isa(obj, Vector{Any})
      return "[" * join([__julia_py_repr(i) for i in obj], ",") * "]"
  elseif isa(obj, Vector{Bool})
    if (length(obj) == 1)
      __julia_py_repr_logical_1(obj)
    else
      return "[" * join([__julia_py_repr_logical_1(i) for i in obj], ",") * "]"
    end
  elseif isa(obj, Int)
    __julia_py_repr_integer_1(obj)
  elseif isa(obj, Complex)
    __julia_py_repr_complex_1(obj)
  elseif isa(obj, Float64)
    __julia_py_repr_double_1(obj)
  elseif isa(obj, String)
    __julia_py_repr_character_1(obj)
  elseif isa(obj, Char)
    __julia_py_repr_character_1(string(obj))
  elseif isa(obj, Bool)
    __julia_py_repr_logical_1(obj)
  elseif startswith(string(typeof(obj)),"DataFrame")
    __julia_py_repr_dataframe(obj)
  elseif startswith(string(typeof(obj)),"NamedArray")
    return __julia_py_repr_namedarray(obj)
  else
    return "'Untransferrable variable'"
  end
end
'''


class sos_Julia:
    background_color = '#ebd8eb'
    supported_kernels = {'Julia': ['julia-?.?']}
    options = {'assignment_pattern': r'^([_A-Za-z0-9\.]+)\s*=.*$'}
    cd_command = 'cd("{dir}")'

    def __init__(self, sos_kernel, kernel_name='julia-0.6'):
        self.sos_kernel = sos_kernel
        self.kernel_name = kernel_name
        self.init_statements = julia_init_statements
        self.loaded = set()

    def load(self, package):
        if package in self.loaded:
            return True
        if package in julia_install_package:
            self.sos_kernel.run_cell(
                julia_install_package[package],
                True,
                False,
                on_error=f'Install of package {package} is not supported.')
            self.loaded.add(package)
            return True
        else:
            self.sos_kernel.warn(
                f'Install of package {package} is not supported.')
            return False


#  support for %get
#
#  Converting a Python object to a julia expression that will be executed
#  by the julia kernel.
#
#

    def _julia_repr(self, obj):
        if isinstance(obj, bool):
            return 'true' if obj else 'false'
        elif isinstance(obj, (int, float)):
            return repr(obj)
        elif isinstance(obj, str):
            # Not using repr() here becasue of the problem of qoutes in Julia.
            return '"""' + obj + '"""'
        elif isinstance(obj, complex):
            return 'complex(' + str(obj.real) + ',' + str(obj.imag) + ')'
        elif isinstance(obj, Sequence):
            if len(obj) == 0:
                return '[]'
            else:
                return '[' + ','.join(self._julia_repr(x) for x in obj) + ']'
        elif obj is None:
            return 'NaN'
        elif isinstance(obj, dict):
            return 'Dict(' + ','.join(
                '"{}" => {}'.format(x, self._julia_repr(y))
                for x, y in obj.items()) + ')'
        elif isinstance(obj, set):
            return 'Set([' + ','.join(self._julia_repr(x) for x in obj) + '])'
        else:
            if isinstance(obj, (numpy.intc, numpy.intp, numpy.int8, numpy.int16, numpy.int32, numpy.int64,\
                    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float16, numpy.float32)):
                return repr(obj)
            # need to specify Float64() as the return to Julia in order to avoid losing precision
            elif isinstance(obj, numpy.float64):
                return 'Float64(' + obj + ')'
            elif isinstance(obj, numpy.matrixlib.defmatrix.matrix):
                try:
                    import feather
                except ImportError:
                    raise UsageError(
                        'The feather-format module is required to pass numpy matrix as julia matrix(array)'
                        'See https://github.com/wesm/feather/tree/master/python for details.'
                    )
                feather_tmp_ = tempfile.NamedTemporaryFile(
                    suffix='.feather', delete=False).name
                feather.write_dataframe(
                    pandas.DataFrame(obj).copy(), feather_tmp_)
                return 'convert(Matrix, Feather.read("' + feather_tmp_ + '"))'
            elif isinstance(obj, numpy.ndarray):
                return '[' + ','.join(self._julia_repr(x) for x in obj) + ']'
            elif isinstance(obj, pandas.DataFrame):
                try:
                    import feather
                except ImportError:
                    raise UsageError(
                        'The feather-format module is required to pass pandas DataFrame as julia.DataFrames'
                        'See https://github.com/wesm/feather/tree/master/python for details.'
                    )
                feather_tmp_ = tempfile.NamedTemporaryFile(
                    suffix='.feather', delete=False).name
                try:
                    data = obj.copy()
                    # Julia DataFrame does not have index
                    if not isinstance(data.index, pandas.RangeIndex):
                        self.sos_kernel.warn(
                            'Raw index is ignored because Julia DataFrame does not support raw index.'
                        )
                    feather.write_dataframe(data, feather_tmp_)
                except Exception:
                    # if data cannot be written, we try to manipulate data
                    # frame to have consistent types and try again
                    for c in data.columns:
                        if not homogeneous_type(data[c]):
                            data[c] = [str(x) for x in data[c]]
                    feather.write_dataframe(data, feather_tmp_)
                    # use {!r} for path because the string might contain c:\ which needs to be
                    # double quoted.
                return 'Feather.read("' + feather_tmp_ + '")'
            elif isinstance(obj, pandas.Series):
                dat = list(obj.values)
                ind = list(obj.index.values)
                ans = 'NamedArray(' + '[' + ','.join(
                    self._julia_repr(x) for x in dat) + ']' + ',([' + ','.join(
                        self._julia_repr(y) for y in ind) + '],))'
                return ans.replace("'", '"')
            else:
                return repr('Unsupported datatype {}'.format(short_repr(obj)))

    def get_vars(self, names):
        for name in names:
            if name.startswith('_'):
                self.sos_kernel.warn(
                    'Variable {} is passed from SoS to kernel {} as {}'.format(
                        name, self.kernel_name, '.' + name[1:]))
                newname = '.' + name[1:]
            else:
                newname = name
            julia_repr = self._julia_repr(env.sos_dict[name])
            if 'Feather.' in julia_repr:
                self.load('feather')
            if 'NamedArray' in julia_repr:
                self.load('namedarray')
            if 'DataFrame' in julia_repr:
                self.load('dataframes')
            self.sos_kernel.run_cell(
                '{} = {}'.format(newname, julia_repr),
                True,
                False,
                on_error='Failed to put variable {} to julia'.format(name))

    def put_vars(self, items, to_kernel=None, as_type=None):
        # first let us get all variables with names starting with sos
        try:
            response = self.sos_kernel.get_response(
                'print(string(varinfo(r"sos")))', ('stream',),
                name=('stdout',))[0][1]
            all_markdown_items = [
                x.strip() for x in response['text'].split('|')
            ]
            items += [x for x in all_markdown_items if x.startswith('sos')]
        except:
            # if there is no variable with name sos, the command will not produce any output
            pass

        if not items:
            return {}

        res = {}
        for item in items:
            while True:
                #
                # Issue #3: Beause it takes a long time to load Julia Module, we do not import
                # dataframe, namedarray etc when Julia starts. Rather, we generate an error
                # message if thosse packages are needed and "using" or "Add" them when needed.
                #
                py_repr = '__julia_py_repr({})'.format(item)
                response = self.sos_kernel.get_response(
                    py_repr, ('execute_result',))[0][1]
                expr = response['data']['text/plain']

                if expr.startswith('"SOS_JULIA_REQUIRE:'):
                    package = expr.split(':')[1].rstrip('"')
                    if not self.load(package):
                        break
                else:
                    break

            try:
                if 'read_dataframe' in expr:
                    # imported to be used by eval
                    from feather import read_dataframe
                    # suppress flakes warning
                    read_dataframe
                # evaluate as raw string to correctly handle \\ etc
                res[item] = eval(eval(expr))
            except Exception as e:
                self.sos_kernel.warn('Failed to evaluate {!r}: {}'.format(
                    expr, e))
                return None
        return res

    def sessioninfo(self):
        ret = self.sos_kernel.get_response(
            r'versioninfo(true)', ('stream',), name=('stdout',))
        return '\n'.join(x[1]['text'] for x in ret)
