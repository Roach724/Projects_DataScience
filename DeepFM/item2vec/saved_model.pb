֮
�'�'
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype�
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype�
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
�
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint���������
;
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8�
�
tag_embedding_layer/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:3*/
shared_name tag_embedding_layer/embeddings
�
2tag_embedding_layer/embeddings/Read/ReadVariableOpReadVariableOptag_embedding_layer/embeddings*
_output_shapes

:3*
dtype0
�
id_embedding_layer/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:e*.
shared_nameid_embedding_layer/embeddings
�
1id_embedding_layer/embeddings/Read/ReadVariableOpReadVariableOpid_embedding_layer/embeddings*
_output_shapes

:e*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_2*
value_dtype0	
�
string_lookup_1_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_9*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
�
%Adam/tag_embedding_layer/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:3*6
shared_name'%Adam/tag_embedding_layer/embeddings/m
�
9Adam/tag_embedding_layer/embeddings/m/Read/ReadVariableOpReadVariableOp%Adam/tag_embedding_layer/embeddings/m*
_output_shapes

:3*
dtype0
�
$Adam/id_embedding_layer/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:e*5
shared_name&$Adam/id_embedding_layer/embeddings/m
�
8Adam/id_embedding_layer/embeddings/m/Read/ReadVariableOpReadVariableOp$Adam/id_embedding_layer/embeddings/m*
_output_shapes

:e*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
�
%Adam/tag_embedding_layer/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:3*6
shared_name'%Adam/tag_embedding_layer/embeddings/v
�
9Adam/tag_embedding_layer/embeddings/v/Read/ReadVariableOpReadVariableOp%Adam/tag_embedding_layer/embeddings/v*
_output_shapes

:3*
dtype0
�
$Adam/id_embedding_layer/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:e*5
shared_name&$Adam/id_embedding_layer/embeddings/v
�
8Adam/id_embedding_layer/embeddings/v/Read/ReadVariableOpReadVariableOp$Adam/id_embedding_layer/embeddings/v*
_output_shapes

:e*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R
�
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *"
fR
__inference_<lambda>_3815
�
PartitionedCall_1PartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *"
fR
__inference_<lambda>_3820
2
NoOpNoOp^PartitionedCall^PartitionedCall_1
�
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_index_table*
Tkeys0*
Tvalues0	*,
_class"
 loc:@string_lookup_index_table*
_output_shapes

::
�
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_1_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_1_index_table*
_output_shapes

::
�1
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�1
value�1B�1 B�1
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
=
state_variables
_index_lookup_layer
	keras_api
=
state_variables
_index_lookup_layer
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
 	keras_api

!	keras_api
R
"	variables
#trainable_variables
$regularization_losses
%	keras_api
h

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
�
2iter

3beta_1

4beta_2
	5decay
6learning_ratemgmh&mi'mj,mk-mlvmvn&vo'vp,vq-vr
*
2
3
&4
'5
,6
-7
*
0
1
&2
'3
,4
-5
 
�
7metrics
8layer_metrics
	variables
9non_trainable_variables

:layers
trainable_variables
regularization_losses
;layer_regularization_losses
 
 
0
<state_variables

=_table
>	keras_api
 
 
0
?state_variables

@_table
A	keras_api
 
nl
VARIABLE_VALUEtag_embedding_layer/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
Bmetrics
Clayer_metrics
	variables
Dnon_trainable_variables

Elayers
trainable_variables
regularization_losses
Flayer_regularization_losses
mk
VARIABLE_VALUEid_embedding_layer/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
Gmetrics
Hlayer_metrics
	variables
Inon_trainable_variables

Jlayers
trainable_variables
regularization_losses
Klayer_regularization_losses
 
 
 
 
�
Lmetrics
Mlayer_metrics
"	variables
Nnon_trainable_variables

Olayers
#trainable_variables
$regularization_losses
Player_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
�
Qmetrics
Rlayer_metrics
(	variables
Snon_trainable_variables

Tlayers
)trainable_variables
*regularization_losses
Ulayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
�
Vmetrics
Wlayer_metrics
.	variables
Xnon_trainable_variables

Ylayers
/trainable_variables
0regularization_losses
Zlayer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
 
 
F
0
1
2
3
4
5
6
7
	8

9
 
 
LJ
tableAlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table
 
 
LJ
tableAlayer_with_weights-1/_index_lookup_layer/_table/.ATTRIBUTES/table
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	]total
	^count
_	variables
`	keras_api
p
atrue_positives
btrue_negatives
cfalse_positives
dfalse_negatives
e	variables
f	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

_	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

a0
b1
c2
d3

e	variables
��
VARIABLE_VALUE%Adam/tag_embedding_layer/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE$Adam/id_embedding_layer/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%Adam/tag_embedding_layer/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE$Adam/id_embedding_layer/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
u
serving_default_idPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
{
serving_default_item_tagPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_idserving_default_item_tagstring_lookup_1_index_tableConststring_lookup_index_tableConst_1tag_embedding_layer/embeddingsid_embedding_layer/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference_signature_wrapper_3160
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2tag_embedding_layer/embeddings/Read/ReadVariableOp1id_embedding_layer/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpHstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:1Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp9Adam/tag_embedding_layer/embeddings/m/Read/ReadVariableOp8Adam/id_embedding_layer/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp9Adam/tag_embedding_layer/embeddings/v/Read/ReadVariableOp8Adam/id_embedding_layer/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst_2*.
Tin'
%2#			*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference__traced_save_3945
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametag_embedding_layer/embeddingsid_embedding_layer/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratestring_lookup_index_tablestring_lookup_1_index_tabletotalcounttrue_positivestrue_negativesfalse_positivesfalse_negatives%Adam/tag_embedding_layer/embeddings/m$Adam/id_embedding_layer/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/m%Adam/tag_embedding_layer/embeddings/v$Adam/id_embedding_layer/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_restore_4048��
�
�
M__inference_tag_embedding_layer_layer_call_and_return_conditional_losses_3641

inputs	
embedding_lookup_3635
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_3635inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/3635*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/3635*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
id_vectorize_cond_true_29855
1id_vectorize_cond_pad_paddings_1_id_vectorize_subJ
Fid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
"id_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2$
"id_vectorize/cond/Pad/paddings/1/0�
 id_vectorize/cond/Pad/paddings/1Pack+id_vectorize/cond/Pad/paddings/1/0:output:01id_vectorize_cond_pad_paddings_1_id_vectorize_sub*
N*
T0*
_output_shapes
:2"
 id_vectorize/cond/Pad/paddings/1�
"id_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"id_vectorize/cond/Pad/paddings/0_1�
id_vectorize/cond/Pad/paddingsPack+id_vectorize/cond/Pad/paddings/0_1:output:0)id_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2 
id_vectorize/cond/Pad/paddings�
id_vectorize/cond/PadPadFid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor'id_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Pad�
id_vectorize/cond/IdentityIdentityid_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
S
7__inference_global_average_pooling1d_layer_call_fn_3686

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_22802
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
��
�
?__inference_model_layer_call_and_return_conditional_losses_3397
	inputs_id
inputs_item_tagV
Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleW
Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	U
Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleV
Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	-
)tag_embedding_layer_embedding_lookup_3368,
(id_embedding_layer_embedding_lookup_3373(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�#id_embedding_layer/embedding_lookup�Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2�$tag_embedding_layer/embedding_lookup�Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2w
id_vectorize/StringLowerStringLower	inputs_id*'
_output_shapes
:���������2
id_vectorize/StringLower�
id_vectorize/StaticRegexReplaceStaticRegexReplace!id_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2!
id_vectorize/StaticRegexReplace�
id_vectorize/SqueezeSqueeze(id_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
id_vectorize/Squeeze�
id_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2 
id_vectorize/StringSplit/Const�
&id_vectorize/StringSplit/StringSplitV2StringSplitV2id_vectorize/Squeeze:output:0'id_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2(
&id_vectorize/StringSplit/StringSplitV2�
,id_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,id_vectorize/StringSplit/strided_slice/stack�
.id_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.id_vectorize/StringSplit/strided_slice/stack_1�
.id_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.id_vectorize/StringSplit/strided_slice/stack_2�
&id_vectorize/StringSplit/strided_sliceStridedSlice0id_vectorize/StringSplit/StringSplitV2:indices:05id_vectorize/StringSplit/strided_slice/stack:output:07id_vectorize/StringSplit/strided_slice/stack_1:output:07id_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2(
&id_vectorize/StringSplit/strided_slice�
.id_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.id_vectorize/StringSplit/strided_slice_1/stack�
0id_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_1�
0id_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_2�
(id_vectorize/StringSplit/strided_slice_1StridedSlice.id_vectorize/StringSplit/StringSplitV2:shape:07id_vectorize/StringSplit/strided_slice_1/stack:output:09id_vectorize/StringSplit/strided_slice_1/stack_1:output:09id_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2*
(id_vectorize/StringSplit/strided_slice_1�
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast/id_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2Q
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast1id_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdbid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2_
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateraid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0fid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2`id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2^
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumcid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2\
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2cid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle/id_vectorize/StringSplit/StringSplitV2:values:0Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2G
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2~
.id_vectorize/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 20
.id_vectorize/string_lookup_1/assert_equal/NoOp�
%id_vectorize/string_lookup_1/IdentityIdentityNid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2'
%id_vectorize/string_lookup_1/Identity�
'id_vectorize/string_lookup_1/Identity_1IdentityZid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2)
'id_vectorize/string_lookup_1/Identity_1�
)id_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2+
)id_vectorize/RaggedToTensor/default_value�
!id_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2#
!id_vectorize/RaggedToTensor/Const�
0id_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor*id_vectorize/RaggedToTensor/Const:output:0.id_vectorize/string_lookup_1/Identity:output:02id_vectorize/RaggedToTensor/default_value:output:00id_vectorize/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS22
0id_vectorize/RaggedToTensor/RaggedTensorToTensor�
id_vectorize/ShapeShape9id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
id_vectorize/Shape�
 id_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 id_vectorize/strided_slice/stack�
"id_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_1�
"id_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_2�
id_vectorize/strided_sliceStridedSliceid_vectorize/Shape:output:0)id_vectorize/strided_slice/stack:output:0+id_vectorize/strided_slice/stack_1:output:0+id_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
id_vectorize/strided_slicej
id_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/sub/x�
id_vectorize/subSubid_vectorize/sub/x:output:0#id_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
id_vectorize/subl
id_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/Less/y�
id_vectorize/LessLess#id_vectorize/strided_slice:output:0id_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
id_vectorize/Less�
id_vectorize/condStatelessIfid_vectorize/Less:z:0id_vectorize/sub:z:09id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 */
else_branch R
id_vectorize_cond_false_3273*/
output_shapes
:������������������*.
then_branchR
id_vectorize_cond_true_32722
id_vectorize/cond�
id_vectorize/cond/IdentityIdentityid_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
id_vectorize/cond/Identity
tag_vectorize/StringLowerStringLowerinputs_item_tag*'
_output_shapes
:���������2
tag_vectorize/StringLower�
 tag_vectorize/StaticRegexReplaceStaticRegexReplace"tag_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2"
 tag_vectorize/StaticRegexReplace�
tag_vectorize/SqueezeSqueeze)tag_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
tag_vectorize/Squeeze�
tag_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2!
tag_vectorize/StringSplit/Const�
'tag_vectorize/StringSplit/StringSplitV2StringSplitV2tag_vectorize/Squeeze:output:0(tag_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2)
'tag_vectorize/StringSplit/StringSplitV2�
-tag_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tag_vectorize/StringSplit/strided_slice/stack�
/tag_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/tag_vectorize/StringSplit/strided_slice/stack_1�
/tag_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/tag_vectorize/StringSplit/strided_slice/stack_2�
'tag_vectorize/StringSplit/strided_sliceStridedSlice1tag_vectorize/StringSplit/StringSplitV2:indices:06tag_vectorize/StringSplit/strided_slice/stack:output:08tag_vectorize/StringSplit/strided_slice/stack_1:output:08tag_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'tag_vectorize/StringSplit/strided_slice�
/tag_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tag_vectorize/StringSplit/strided_slice_1/stack�
1tag_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_1�
1tag_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_2�
)tag_vectorize/StringSplit/strided_slice_1StridedSlice/tag_vectorize/StringSplit/StringSplitV2:shape:08tag_vectorize/StringSplit/strided_slice_1/stack:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_1:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2+
)tag_vectorize/StringSplit/strided_slice_1�
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast0tag_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2R
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast2tag_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2`
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterbtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0gtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2atag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2_
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumdtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2]
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2dtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle0tag_vectorize/StringSplit/StringSplitV2:values:0Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2F
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2|
-tag_vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2/
-tag_vectorize/string_lookup/assert_equal/NoOp�
$tag_vectorize/string_lookup/IdentityIdentityMtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2&
$tag_vectorize/string_lookup/Identity�
&tag_vectorize/string_lookup/Identity_1Identity[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2(
&tag_vectorize/string_lookup/Identity_1�
*tag_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2,
*tag_vectorize/RaggedToTensor/default_value�
"tag_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2$
"tag_vectorize/RaggedToTensor/Const�
1tag_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor+tag_vectorize/RaggedToTensor/Const:output:0-tag_vectorize/string_lookup/Identity:output:03tag_vectorize/RaggedToTensor/default_value:output:0/tag_vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS23
1tag_vectorize/RaggedToTensor/RaggedTensorToTensor�
tag_vectorize/ShapeShape:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
tag_vectorize/Shape�
!tag_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!tag_vectorize/strided_slice/stack�
#tag_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_1�
#tag_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_2�
tag_vectorize/strided_sliceStridedSlicetag_vectorize/Shape:output:0*tag_vectorize/strided_slice/stack:output:0,tag_vectorize/strided_slice/stack_1:output:0,tag_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
tag_vectorize/strided_slicel
tag_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/sub/x�
tag_vectorize/subSubtag_vectorize/sub/x:output:0$tag_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
tag_vectorize/subn
tag_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/Less/y�
tag_vectorize/LessLess$tag_vectorize/strided_slice:output:0tag_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
tag_vectorize/Less�
tag_vectorize/condStatelessIftag_vectorize/Less:z:0tag_vectorize/sub:z:0:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
else_branch!R
tag_vectorize_cond_false_3348*/
output_shapes
:������������������*/
then_branch R
tag_vectorize_cond_true_33472
tag_vectorize/cond�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
tag_vectorize/cond/Identity�
$tag_embedding_layer/embedding_lookupResourceGather)tag_embedding_layer_embedding_lookup_3368$tag_vectorize/cond/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*<
_class2
0.loc:@tag_embedding_layer/embedding_lookup/3368*+
_output_shapes
:���������*
dtype02&
$tag_embedding_layer/embedding_lookup�
-tag_embedding_layer/embedding_lookup/IdentityIdentity-tag_embedding_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@tag_embedding_layer/embedding_lookup/3368*+
_output_shapes
:���������2/
-tag_embedding_layer/embedding_lookup/Identity�
/tag_embedding_layer/embedding_lookup/Identity_1Identity6tag_embedding_layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������21
/tag_embedding_layer/embedding_lookup/Identity_1�
#id_embedding_layer/embedding_lookupResourceGather(id_embedding_layer_embedding_lookup_3373#id_vectorize/cond/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*;
_class1
/-loc:@id_embedding_layer/embedding_lookup/3373*+
_output_shapes
:���������*
dtype02%
#id_embedding_layer/embedding_lookup�
,id_embedding_layer/embedding_lookup/IdentityIdentity,id_embedding_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@id_embedding_layer/embedding_lookup/3373*+
_output_shapes
:���������2.
,id_embedding_layer/embedding_lookup/Identity�
.id_embedding_layer/embedding_lookup/Identity_1Identity5id_embedding_layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������20
.id_embedding_layer/embedding_lookup/Identity_1p
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis�
tf.concat/concatConcatV28tag_embedding_layer/embedding_lookup/Identity_1:output:07id_embedding_layer/embedding_lookup/Identity_1:output:0tf.concat/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	2
tf.concat/concat�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMeantf.concat/concat:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������2
global_average_pooling1d/Mean�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Sigmoid�
IdentityIdentitydense_1/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp$^id_embedding_layer/embedding_lookupF^id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2%^tag_embedding_layer/embedding_lookupE^tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2J
#id_embedding_layer/embedding_lookup#id_embedding_layer/embedding_lookup2�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV22L
$tag_embedding_layer/embedding_lookup$tag_embedding_layer/embedding_lookup2�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:R N
'
_output_shapes
:���������
#
_user_specified_name	inputs/id:XT
'
_output_shapes
:���������
)
_user_specified_nameinputs/item_tag:

_output_shapes
: :

_output_shapes
: 
�
�
id_vectorize_cond_false_3273!
id_vectorize_cond_placeholderT
Pid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
%id_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%id_vectorize/cond/strided_slice/stack�
'id_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'id_vectorize/cond/strided_slice/stack_1�
'id_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'id_vectorize/cond/strided_slice/stack_2�
id_vectorize/cond/strided_sliceStridedSlicePid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor.id_vectorize/cond/strided_slice/stack:output:00id_vectorize/cond/strided_slice/stack_1:output:00id_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2!
id_vectorize/cond/strided_slice�
id_vectorize/cond/IdentityIdentity(id_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
id_vectorize_cond_true_23425
1id_vectorize_cond_pad_paddings_1_id_vectorize_subJ
Fid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
"id_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2$
"id_vectorize/cond/Pad/paddings/1/0�
 id_vectorize/cond/Pad/paddings/1Pack+id_vectorize/cond/Pad/paddings/1/0:output:01id_vectorize_cond_pad_paddings_1_id_vectorize_sub*
N*
T0*
_output_shapes
:2"
 id_vectorize/cond/Pad/paddings/1�
"id_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"id_vectorize/cond/Pad/paddings/0_1�
id_vectorize/cond/Pad/paddingsPack+id_vectorize/cond/Pad/paddings/0_1:output:0)id_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2 
id_vectorize/cond/Pad/paddings�
id_vectorize/cond/PadPadFid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor'id_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Pad�
id_vectorize/cond/IdentityIdentityid_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�	
�
?__inference_dense_layer_call_and_return_conditional_losses_3697

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
w
1__inference_id_embedding_layer_layer_call_fn_3664

inputs	
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_id_embedding_layer_layer_call_and_return_conditional_losses_24682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
id_vectorize_cond_false_2608!
id_vectorize_cond_placeholderT
Pid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
%id_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%id_vectorize/cond/strided_slice/stack�
'id_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'id_vectorize/cond/strided_slice/stack_1�
'id_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'id_vectorize/cond/strided_slice/stack_2�
id_vectorize/cond/strided_sliceStridedSlicePid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor.id_vectorize/cond/strided_slice/stack:output:00id_vectorize/cond/strided_slice/stack_1:output:00id_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2!
id_vectorize/cond/strided_slice�
id_vectorize/cond/IdentityIdentity(id_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�

�
$__inference_model_layer_call_fn_3632
	inputs_id
inputs_item_tag
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	inputs_idinputs_item_tagunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_31012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������
#
_user_specified_name	inputs/id:XT
'
_output_shapes
:���������
)
_user_specified_nameinputs/item_tag:

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_model_layer_call_and_return_conditional_losses_2901

inputs
inputs_1V
Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleW
Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	U
Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleV
Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
tag_embedding_layer_2881
id_embedding_layer_2884

dense_2890

dense_2892
dense_1_2895
dense_1_2897
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�*id_embedding_layer/StatefulPartitionedCall�Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2�+tag_embedding_layer/StatefulPartitionedCall�Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2t
id_vectorize/StringLowerStringLowerinputs*'
_output_shapes
:���������2
id_vectorize/StringLower�
id_vectorize/StaticRegexReplaceStaticRegexReplace!id_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2!
id_vectorize/StaticRegexReplace�
id_vectorize/SqueezeSqueeze(id_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
id_vectorize/Squeeze�
id_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2 
id_vectorize/StringSplit/Const�
&id_vectorize/StringSplit/StringSplitV2StringSplitV2id_vectorize/Squeeze:output:0'id_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2(
&id_vectorize/StringSplit/StringSplitV2�
,id_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,id_vectorize/StringSplit/strided_slice/stack�
.id_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.id_vectorize/StringSplit/strided_slice/stack_1�
.id_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.id_vectorize/StringSplit/strided_slice/stack_2�
&id_vectorize/StringSplit/strided_sliceStridedSlice0id_vectorize/StringSplit/StringSplitV2:indices:05id_vectorize/StringSplit/strided_slice/stack:output:07id_vectorize/StringSplit/strided_slice/stack_1:output:07id_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2(
&id_vectorize/StringSplit/strided_slice�
.id_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.id_vectorize/StringSplit/strided_slice_1/stack�
0id_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_1�
0id_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_2�
(id_vectorize/StringSplit/strided_slice_1StridedSlice.id_vectorize/StringSplit/StringSplitV2:shape:07id_vectorize/StringSplit/strided_slice_1/stack:output:09id_vectorize/StringSplit/strided_slice_1/stack_1:output:09id_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2*
(id_vectorize/StringSplit/strided_slice_1�
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast/id_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2Q
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast1id_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdbid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2_
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateraid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0fid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2`id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2^
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumcid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2\
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2cid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle/id_vectorize/StringSplit/StringSplitV2:values:0Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2G
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2~
.id_vectorize/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 20
.id_vectorize/string_lookup_1/assert_equal/NoOp�
%id_vectorize/string_lookup_1/IdentityIdentityNid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2'
%id_vectorize/string_lookup_1/Identity�
'id_vectorize/string_lookup_1/Identity_1IdentityZid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2)
'id_vectorize/string_lookup_1/Identity_1�
)id_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2+
)id_vectorize/RaggedToTensor/default_value�
!id_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2#
!id_vectorize/RaggedToTensor/Const�
0id_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor*id_vectorize/RaggedToTensor/Const:output:0.id_vectorize/string_lookup_1/Identity:output:02id_vectorize/RaggedToTensor/default_value:output:00id_vectorize/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS22
0id_vectorize/RaggedToTensor/RaggedTensorToTensor�
id_vectorize/ShapeShape9id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
id_vectorize/Shape�
 id_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 id_vectorize/strided_slice/stack�
"id_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_1�
"id_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_2�
id_vectorize/strided_sliceStridedSliceid_vectorize/Shape:output:0)id_vectorize/strided_slice/stack:output:0+id_vectorize/strided_slice/stack_1:output:0+id_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
id_vectorize/strided_slicej
id_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/sub/x�
id_vectorize/subSubid_vectorize/sub/x:output:0#id_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
id_vectorize/subl
id_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/Less/y�
id_vectorize/LessLess#id_vectorize/strided_slice:output:0id_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
id_vectorize/Less�
id_vectorize/condStatelessIfid_vectorize/Less:z:0id_vectorize/sub:z:09id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 */
else_branch R
id_vectorize_cond_false_2786*/
output_shapes
:������������������*.
then_branchR
id_vectorize_cond_true_27852
id_vectorize/cond�
id_vectorize/cond/IdentityIdentityid_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
id_vectorize/cond/Identityx
tag_vectorize/StringLowerStringLowerinputs_1*'
_output_shapes
:���������2
tag_vectorize/StringLower�
 tag_vectorize/StaticRegexReplaceStaticRegexReplace"tag_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2"
 tag_vectorize/StaticRegexReplace�
tag_vectorize/SqueezeSqueeze)tag_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
tag_vectorize/Squeeze�
tag_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2!
tag_vectorize/StringSplit/Const�
'tag_vectorize/StringSplit/StringSplitV2StringSplitV2tag_vectorize/Squeeze:output:0(tag_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2)
'tag_vectorize/StringSplit/StringSplitV2�
-tag_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tag_vectorize/StringSplit/strided_slice/stack�
/tag_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/tag_vectorize/StringSplit/strided_slice/stack_1�
/tag_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/tag_vectorize/StringSplit/strided_slice/stack_2�
'tag_vectorize/StringSplit/strided_sliceStridedSlice1tag_vectorize/StringSplit/StringSplitV2:indices:06tag_vectorize/StringSplit/strided_slice/stack:output:08tag_vectorize/StringSplit/strided_slice/stack_1:output:08tag_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'tag_vectorize/StringSplit/strided_slice�
/tag_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tag_vectorize/StringSplit/strided_slice_1/stack�
1tag_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_1�
1tag_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_2�
)tag_vectorize/StringSplit/strided_slice_1StridedSlice/tag_vectorize/StringSplit/StringSplitV2:shape:08tag_vectorize/StringSplit/strided_slice_1/stack:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_1:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2+
)tag_vectorize/StringSplit/strided_slice_1�
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast0tag_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2R
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast2tag_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2`
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterbtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0gtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2atag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2_
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumdtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2]
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2dtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle0tag_vectorize/StringSplit/StringSplitV2:values:0Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2F
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2|
-tag_vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2/
-tag_vectorize/string_lookup/assert_equal/NoOp�
$tag_vectorize/string_lookup/IdentityIdentityMtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2&
$tag_vectorize/string_lookup/Identity�
&tag_vectorize/string_lookup/Identity_1Identity[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2(
&tag_vectorize/string_lookup/Identity_1�
*tag_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2,
*tag_vectorize/RaggedToTensor/default_value�
"tag_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2$
"tag_vectorize/RaggedToTensor/Const�
1tag_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor+tag_vectorize/RaggedToTensor/Const:output:0-tag_vectorize/string_lookup/Identity:output:03tag_vectorize/RaggedToTensor/default_value:output:0/tag_vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS23
1tag_vectorize/RaggedToTensor/RaggedTensorToTensor�
tag_vectorize/ShapeShape:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
tag_vectorize/Shape�
!tag_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!tag_vectorize/strided_slice/stack�
#tag_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_1�
#tag_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_2�
tag_vectorize/strided_sliceStridedSlicetag_vectorize/Shape:output:0*tag_vectorize/strided_slice/stack:output:0,tag_vectorize/strided_slice/stack_1:output:0,tag_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
tag_vectorize/strided_slicel
tag_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/sub/x�
tag_vectorize/subSubtag_vectorize/sub/x:output:0$tag_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
tag_vectorize/subn
tag_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/Less/y�
tag_vectorize/LessLess$tag_vectorize/strided_slice:output:0tag_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
tag_vectorize/Less�
tag_vectorize/condStatelessIftag_vectorize/Less:z:0tag_vectorize/sub:z:0:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
else_branch!R
tag_vectorize_cond_false_2861*/
output_shapes
:������������������*/
then_branch R
tag_vectorize_cond_true_28602
tag_vectorize/cond�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
tag_vectorize/cond/Identity�
+tag_embedding_layer/StatefulPartitionedCallStatefulPartitionedCall$tag_vectorize/cond/Identity:output:0tag_embedding_layer_2881*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_tag_embedding_layer_layer_call_and_return_conditional_losses_24472-
+tag_embedding_layer/StatefulPartitionedCall�
*id_embedding_layer/StatefulPartitionedCallStatefulPartitionedCall#id_vectorize/cond/Identity:output:0id_embedding_layer_2884*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_id_embedding_layer_layer_call_and_return_conditional_losses_24682,
*id_embedding_layer/StatefulPartitionedCallp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis�
tf.concat/concatConcatV24tag_embedding_layer/StatefulPartitionedCall:output:03id_embedding_layer/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	2
tf.concat/concat�
(global_average_pooling1d/PartitionedCallPartitionedCalltf.concat/concat:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_24872*
(global_average_pooling1d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0
dense_2890
dense_2892*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_25052
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2895dense_1_2897*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_25322!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall+^id_embedding_layer/StatefulPartitionedCallF^id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2,^tag_embedding_layer/StatefulPartitionedCallE^tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*id_embedding_layer/StatefulPartitionedCall*id_embedding_layer/StatefulPartitionedCall2�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV22Z
+tag_embedding_layer/StatefulPartitionedCall+tag_embedding_layer/StatefulPartitionedCall2�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2487

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������	:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
A__inference_dense_1_layer_call_and_return_conditional_losses_3717

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
!model_id_vectorize_cond_true_2139A
=model_id_vectorize_cond_pad_paddings_1_model_id_vectorize_subV
Rmodel_id_vectorize_cond_pad_model_id_vectorize_raggedtotensor_raggedtensortotensor	$
 model_id_vectorize_cond_identity	�
(model/id_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2*
(model/id_vectorize/cond/Pad/paddings/1/0�
&model/id_vectorize/cond/Pad/paddings/1Pack1model/id_vectorize/cond/Pad/paddings/1/0:output:0=model_id_vectorize_cond_pad_paddings_1_model_id_vectorize_sub*
N*
T0*
_output_shapes
:2(
&model/id_vectorize/cond/Pad/paddings/1�
(model/id_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2*
(model/id_vectorize/cond/Pad/paddings/0_1�
$model/id_vectorize/cond/Pad/paddingsPack1model/id_vectorize/cond/Pad/paddings/0_1:output:0/model/id_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2&
$model/id_vectorize/cond/Pad/paddings�
model/id_vectorize/cond/PadPadRmodel_id_vectorize_cond_pad_model_id_vectorize_raggedtotensor_raggedtensortotensor-model/id_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
model/id_vectorize/cond/Pad�
 model/id_vectorize/cond/IdentityIdentity$model/id_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2"
 model/id_vectorize/cond/Identity"M
 model_id_vectorize_cond_identity)model/id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�	
�
__inference_restore_fn_3783
restored_tensors_0
restored_tensors_1	L
Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handle
identity��;string_lookup_index_table_table_restore/LookupTableImportV2�
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const�
IdentityIdentityConst:output:0<^string_lookup_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:::2z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
�
�
"model_id_vectorize_cond_false_2140'
#model_id_vectorize_cond_placeholder`
\model_id_vectorize_cond_strided_slice_model_id_vectorize_raggedtotensor_raggedtensortotensor	$
 model_id_vectorize_cond_identity	�
+model/id_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+model/id_vectorize/cond/strided_slice/stack�
-model/id_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model/id_vectorize/cond/strided_slice/stack_1�
-model/id_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-model/id_vectorize/cond/strided_slice/stack_2�
%model/id_vectorize/cond/strided_sliceStridedSlice\model_id_vectorize_cond_strided_slice_model_id_vectorize_raggedtotensor_raggedtensortotensor4model/id_vectorize/cond/strided_slice/stack:output:06model/id_vectorize/cond/strided_slice/stack_1:output:06model/id_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2'
%model/id_vectorize/cond/strided_slice�
 model/id_vectorize/cond/IdentityIdentity.model/id_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2"
 model/id_vectorize/cond/Identity"M
 model_id_vectorize_cond_identity)model/id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
"model_tag_vectorize_cond_true_2214C
?model_tag_vectorize_cond_pad_paddings_1_model_tag_vectorize_subX
Tmodel_tag_vectorize_cond_pad_model_tag_vectorize_raggedtotensor_raggedtensortotensor	%
!model_tag_vectorize_cond_identity	�
)model/tag_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2+
)model/tag_vectorize/cond/Pad/paddings/1/0�
'model/tag_vectorize/cond/Pad/paddings/1Pack2model/tag_vectorize/cond/Pad/paddings/1/0:output:0?model_tag_vectorize_cond_pad_paddings_1_model_tag_vectorize_sub*
N*
T0*
_output_shapes
:2)
'model/tag_vectorize/cond/Pad/paddings/1�
)model/tag_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)model/tag_vectorize/cond/Pad/paddings/0_1�
%model/tag_vectorize/cond/Pad/paddingsPack2model/tag_vectorize/cond/Pad/paddings/0_1:output:00model/tag_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2'
%model/tag_vectorize/cond/Pad/paddings�
model/tag_vectorize/cond/PadPadTmodel_tag_vectorize_cond_pad_model_tag_vectorize_raggedtotensor_raggedtensortotensor.model/tag_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
model/tag_vectorize/cond/Pad�
!model/tag_vectorize/cond/IdentityIdentity%model/tag_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2#
!model/tag_vectorize/cond/Identity"O
!model_tag_vectorize_cond_identity*model/tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
M__inference_tag_embedding_layer_layer_call_and_return_conditional_losses_2447

inputs	
embedding_lookup_2441
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_2441inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/2441*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/2441*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
tag_vectorize_cond_false_2418"
tag_vectorize_cond_placeholderV
Rtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
&tag_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&tag_vectorize/cond/strided_slice/stack�
(tag_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(tag_vectorize/cond/strided_slice/stack_1�
(tag_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(tag_vectorize/cond/strided_slice/stack_2�
 tag_vectorize/cond/strided_sliceStridedSliceRtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor/tag_vectorize/cond/strided_slice/stack:output:01tag_vectorize/cond/strided_slice/stack_1:output:01tag_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2"
 tag_vectorize/cond/strided_slice�
tag_vectorize/cond/IdentityIdentity)tag_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
{
&__inference_dense_1_layer_call_fn_3726

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_25322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
tag_vectorize_cond_true_30607
3tag_vectorize_cond_pad_paddings_1_tag_vectorize_subL
Htag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
#tag_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2%
#tag_vectorize/cond/Pad/paddings/1/0�
!tag_vectorize/cond/Pad/paddings/1Pack,tag_vectorize/cond/Pad/paddings/1/0:output:03tag_vectorize_cond_pad_paddings_1_tag_vectorize_sub*
N*
T0*
_output_shapes
:2#
!tag_vectorize/cond/Pad/paddings/1�
#tag_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#tag_vectorize/cond/Pad/paddings/0_1�
tag_vectorize/cond/Pad/paddingsPack,tag_vectorize/cond/Pad/paddings/0_1:output:0*tag_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2!
tag_vectorize/cond/Pad/paddings�
tag_vectorize/cond/PadPadHtag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor(tag_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Pad�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
-
__inference__initializer_3751
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
�
+
__inference__destroyer_3756
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
ۆ
�
?__inference_model_layer_call_and_return_conditional_losses_2723
id
item_tagV
Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleW
Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	U
Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleV
Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
tag_embedding_layer_2703
id_embedding_layer_2706

dense_2712

dense_2714
dense_1_2717
dense_1_2719
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�*id_embedding_layer/StatefulPartitionedCall�Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2�+tag_embedding_layer/StatefulPartitionedCall�Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2p
id_vectorize/StringLowerStringLowerid*'
_output_shapes
:���������2
id_vectorize/StringLower�
id_vectorize/StaticRegexReplaceStaticRegexReplace!id_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2!
id_vectorize/StaticRegexReplace�
id_vectorize/SqueezeSqueeze(id_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
id_vectorize/Squeeze�
id_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2 
id_vectorize/StringSplit/Const�
&id_vectorize/StringSplit/StringSplitV2StringSplitV2id_vectorize/Squeeze:output:0'id_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2(
&id_vectorize/StringSplit/StringSplitV2�
,id_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,id_vectorize/StringSplit/strided_slice/stack�
.id_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.id_vectorize/StringSplit/strided_slice/stack_1�
.id_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.id_vectorize/StringSplit/strided_slice/stack_2�
&id_vectorize/StringSplit/strided_sliceStridedSlice0id_vectorize/StringSplit/StringSplitV2:indices:05id_vectorize/StringSplit/strided_slice/stack:output:07id_vectorize/StringSplit/strided_slice/stack_1:output:07id_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2(
&id_vectorize/StringSplit/strided_slice�
.id_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.id_vectorize/StringSplit/strided_slice_1/stack�
0id_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_1�
0id_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_2�
(id_vectorize/StringSplit/strided_slice_1StridedSlice.id_vectorize/StringSplit/StringSplitV2:shape:07id_vectorize/StringSplit/strided_slice_1/stack:output:09id_vectorize/StringSplit/strided_slice_1/stack_1:output:09id_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2*
(id_vectorize/StringSplit/strided_slice_1�
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast/id_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2Q
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast1id_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdbid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2_
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateraid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0fid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2`id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2^
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumcid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2\
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2cid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle/id_vectorize/StringSplit/StringSplitV2:values:0Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2G
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2~
.id_vectorize/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 20
.id_vectorize/string_lookup_1/assert_equal/NoOp�
%id_vectorize/string_lookup_1/IdentityIdentityNid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2'
%id_vectorize/string_lookup_1/Identity�
'id_vectorize/string_lookup_1/Identity_1IdentityZid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2)
'id_vectorize/string_lookup_1/Identity_1�
)id_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2+
)id_vectorize/RaggedToTensor/default_value�
!id_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2#
!id_vectorize/RaggedToTensor/Const�
0id_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor*id_vectorize/RaggedToTensor/Const:output:0.id_vectorize/string_lookup_1/Identity:output:02id_vectorize/RaggedToTensor/default_value:output:00id_vectorize/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS22
0id_vectorize/RaggedToTensor/RaggedTensorToTensor�
id_vectorize/ShapeShape9id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
id_vectorize/Shape�
 id_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 id_vectorize/strided_slice/stack�
"id_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_1�
"id_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_2�
id_vectorize/strided_sliceStridedSliceid_vectorize/Shape:output:0)id_vectorize/strided_slice/stack:output:0+id_vectorize/strided_slice/stack_1:output:0+id_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
id_vectorize/strided_slicej
id_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/sub/x�
id_vectorize/subSubid_vectorize/sub/x:output:0#id_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
id_vectorize/subl
id_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/Less/y�
id_vectorize/LessLess#id_vectorize/strided_slice:output:0id_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
id_vectorize/Less�
id_vectorize/condStatelessIfid_vectorize/Less:z:0id_vectorize/sub:z:09id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 */
else_branch R
id_vectorize_cond_false_2608*/
output_shapes
:������������������*.
then_branchR
id_vectorize_cond_true_26072
id_vectorize/cond�
id_vectorize/cond/IdentityIdentityid_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
id_vectorize/cond/Identityx
tag_vectorize/StringLowerStringLoweritem_tag*'
_output_shapes
:���������2
tag_vectorize/StringLower�
 tag_vectorize/StaticRegexReplaceStaticRegexReplace"tag_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2"
 tag_vectorize/StaticRegexReplace�
tag_vectorize/SqueezeSqueeze)tag_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
tag_vectorize/Squeeze�
tag_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2!
tag_vectorize/StringSplit/Const�
'tag_vectorize/StringSplit/StringSplitV2StringSplitV2tag_vectorize/Squeeze:output:0(tag_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2)
'tag_vectorize/StringSplit/StringSplitV2�
-tag_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tag_vectorize/StringSplit/strided_slice/stack�
/tag_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/tag_vectorize/StringSplit/strided_slice/stack_1�
/tag_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/tag_vectorize/StringSplit/strided_slice/stack_2�
'tag_vectorize/StringSplit/strided_sliceStridedSlice1tag_vectorize/StringSplit/StringSplitV2:indices:06tag_vectorize/StringSplit/strided_slice/stack:output:08tag_vectorize/StringSplit/strided_slice/stack_1:output:08tag_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'tag_vectorize/StringSplit/strided_slice�
/tag_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tag_vectorize/StringSplit/strided_slice_1/stack�
1tag_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_1�
1tag_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_2�
)tag_vectorize/StringSplit/strided_slice_1StridedSlice/tag_vectorize/StringSplit/StringSplitV2:shape:08tag_vectorize/StringSplit/strided_slice_1/stack:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_1:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2+
)tag_vectorize/StringSplit/strided_slice_1�
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast0tag_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2R
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast2tag_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2`
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterbtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0gtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2atag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2_
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumdtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2]
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2dtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle0tag_vectorize/StringSplit/StringSplitV2:values:0Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2F
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2|
-tag_vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2/
-tag_vectorize/string_lookup/assert_equal/NoOp�
$tag_vectorize/string_lookup/IdentityIdentityMtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2&
$tag_vectorize/string_lookup/Identity�
&tag_vectorize/string_lookup/Identity_1Identity[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2(
&tag_vectorize/string_lookup/Identity_1�
*tag_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2,
*tag_vectorize/RaggedToTensor/default_value�
"tag_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2$
"tag_vectorize/RaggedToTensor/Const�
1tag_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor+tag_vectorize/RaggedToTensor/Const:output:0-tag_vectorize/string_lookup/Identity:output:03tag_vectorize/RaggedToTensor/default_value:output:0/tag_vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS23
1tag_vectorize/RaggedToTensor/RaggedTensorToTensor�
tag_vectorize/ShapeShape:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
tag_vectorize/Shape�
!tag_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!tag_vectorize/strided_slice/stack�
#tag_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_1�
#tag_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_2�
tag_vectorize/strided_sliceStridedSlicetag_vectorize/Shape:output:0*tag_vectorize/strided_slice/stack:output:0,tag_vectorize/strided_slice/stack_1:output:0,tag_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
tag_vectorize/strided_slicel
tag_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/sub/x�
tag_vectorize/subSubtag_vectorize/sub/x:output:0$tag_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
tag_vectorize/subn
tag_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/Less/y�
tag_vectorize/LessLess$tag_vectorize/strided_slice:output:0tag_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
tag_vectorize/Less�
tag_vectorize/condStatelessIftag_vectorize/Less:z:0tag_vectorize/sub:z:0:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
else_branch!R
tag_vectorize_cond_false_2683*/
output_shapes
:������������������*/
then_branch R
tag_vectorize_cond_true_26822
tag_vectorize/cond�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
tag_vectorize/cond/Identity�
+tag_embedding_layer/StatefulPartitionedCallStatefulPartitionedCall$tag_vectorize/cond/Identity:output:0tag_embedding_layer_2703*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_tag_embedding_layer_layer_call_and_return_conditional_losses_24472-
+tag_embedding_layer/StatefulPartitionedCall�
*id_embedding_layer/StatefulPartitionedCallStatefulPartitionedCall#id_vectorize/cond/Identity:output:0id_embedding_layer_2706*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_id_embedding_layer_layer_call_and_return_conditional_losses_24682,
*id_embedding_layer/StatefulPartitionedCallp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis�
tf.concat/concatConcatV24tag_embedding_layer/StatefulPartitionedCall:output:03id_embedding_layer/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	2
tf.concat/concat�
(global_average_pooling1d/PartitionedCallPartitionedCalltf.concat/concat:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_24872*
(global_average_pooling1d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0
dense_2712
dense_2714*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_25052
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2717dense_1_2719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_25322!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall+^id_embedding_layer/StatefulPartitionedCallF^id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2,^tag_embedding_layer/StatefulPartitionedCallE^tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*id_embedding_layer/StatefulPartitionedCall*id_embedding_layer/StatefulPartitionedCall2�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV22Z
+tag_embedding_layer/StatefulPartitionedCall+tag_embedding_layer/StatefulPartitionedCall2�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:K G
'
_output_shapes
:���������

_user_specified_nameid:QM
'
_output_shapes
:���������
"
_user_specified_name
item_tag:

_output_shapes
: :

_output_shapes
: 
�
�
__inference_save_fn_3802
checkpoint_key[
Wstring_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	��Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2�
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2L
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1�
IdentityIdentityadd:z:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const�

Identity_1IdentityConst:output:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1�

Identity_2IdentityQstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2�

Identity_3Identity	add_1:z:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1�

Identity_4IdentityConst_1:output:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4�

Identity_5IdentitySstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2�
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
�
�
tag_vectorize_cond_false_3531"
tag_vectorize_cond_placeholderV
Rtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
&tag_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&tag_vectorize/cond/strided_slice/stack�
(tag_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(tag_vectorize/cond/strided_slice/stack_1�
(tag_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(tag_vectorize/cond/strided_slice/stack_2�
 tag_vectorize/cond/strided_sliceStridedSliceRtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor/tag_vectorize/cond/strided_slice/stack:output:01tag_vectorize/cond/strided_slice/stack_1:output:01tag_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2"
 tag_vectorize/cond/strided_slice�
tag_vectorize/cond/IdentityIdentity)tag_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�	
�
$__inference_model_layer_call_fn_2924
id
item_tag
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalliditem_tagunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_29012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
'
_output_shapes
:���������

_user_specified_nameid:QM
'
_output_shapes
:���������
"
_user_specified_name
item_tag:

_output_shapes
: :

_output_shapes
: 
�	
�
$__inference_model_layer_call_fn_3124
id
item_tag
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalliditem_tagunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_31012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
'
_output_shapes
:���������

_user_specified_nameid:QM
'
_output_shapes
:���������
"
_user_specified_name
item_tag:

_output_shapes
: :

_output_shapes
: 
�
�
tag_vectorize_cond_true_26827
3tag_vectorize_cond_pad_paddings_1_tag_vectorize_subL
Htag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
#tag_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2%
#tag_vectorize/cond/Pad/paddings/1/0�
!tag_vectorize/cond/Pad/paddings/1Pack,tag_vectorize/cond/Pad/paddings/1/0:output:03tag_vectorize_cond_pad_paddings_1_tag_vectorize_sub*
N*
T0*
_output_shapes
:2#
!tag_vectorize/cond/Pad/paddings/1�
#tag_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#tag_vectorize/cond/Pad/paddings/0_1�
tag_vectorize/cond/Pad/paddingsPack,tag_vectorize/cond/Pad/paddings/0_1:output:0*tag_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2!
tag_vectorize/cond/Pad/paddings�
tag_vectorize/cond/PadPadHtag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor(tag_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Pad�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
tag_vectorize_cond_false_2861"
tag_vectorize_cond_placeholderV
Rtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
&tag_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&tag_vectorize/cond/strided_slice/stack�
(tag_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(tag_vectorize/cond/strided_slice/stack_1�
(tag_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(tag_vectorize/cond/strided_slice/stack_2�
 tag_vectorize/cond/strided_sliceStridedSliceRtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor/tag_vectorize/cond/strided_slice/stack:output:01tag_vectorize/cond/strided_slice/stack_1:output:01tag_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2"
 tag_vectorize/cond/strided_slice�
tag_vectorize/cond/IdentityIdentity)tag_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
tag_vectorize_cond_false_2683"
tag_vectorize_cond_placeholderV
Rtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
&tag_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&tag_vectorize/cond/strided_slice/stack�
(tag_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(tag_vectorize/cond/strided_slice/stack_1�
(tag_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(tag_vectorize/cond/strided_slice/stack_2�
 tag_vectorize/cond/strided_sliceStridedSliceRtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor/tag_vectorize/cond/strided_slice/stack:output:01tag_vectorize/cond/strided_slice/stack_1:output:01tag_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2"
 tag_vectorize/cond/strided_slice�
tag_vectorize/cond/IdentityIdentity)tag_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
tag_vectorize_cond_true_35307
3tag_vectorize_cond_pad_paddings_1_tag_vectorize_subL
Htag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
#tag_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2%
#tag_vectorize/cond/Pad/paddings/1/0�
!tag_vectorize/cond/Pad/paddings/1Pack,tag_vectorize/cond/Pad/paddings/1/0:output:03tag_vectorize_cond_pad_paddings_1_tag_vectorize_sub*
N*
T0*
_output_shapes
:2#
!tag_vectorize/cond/Pad/paddings/1�
#tag_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#tag_vectorize/cond/Pad/paddings/0_1�
tag_vectorize/cond/Pad/paddingsPack,tag_vectorize/cond/Pad/paddings/0_1:output:0*tag_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2!
tag_vectorize/cond/Pad/paddings�
tag_vectorize/cond/PadPadHtag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor(tag_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Pad�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�K
�
__inference__traced_save_3945
file_prefix=
9savev2_tag_embedding_layer_embeddings_read_readvariableop<
8savev2_id_embedding_layer_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopS
Osavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2U
Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1	U
Qsavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableopD
@savev2_adam_tag_embedding_layer_embeddings_m_read_readvariableopC
?savev2_adam_id_embedding_layer_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopD
@savev2_adam_tag_embedding_layer_embeddings_v_read_readvariableopC
?savev2_adam_id_embedding_layer_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const_2

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*�
value�B�"B:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesBFlayer_with_weights-1/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-1/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_tag_embedding_layer_embeddings_read_readvariableop8savev2_id_embedding_layer_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopOsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1Qsavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop@savev2_adam_tag_embedding_layer_embeddings_m_read_readvariableop?savev2_adam_id_embedding_layer_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop@savev2_adam_tag_embedding_layer_embeddings_v_read_readvariableop?savev2_adam_id_embedding_layer_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"			2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :3:e::::: : : : : ::::: : :�:�:�:�:3:e:::::3:e::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:3:$ 

_output_shapes

:e:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:$ 

_output_shapes

:3:$ 

_output_shapes

:e:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:3:$ 

_output_shapes

:e:$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::"

_output_shapes
: 
�
�
tag_vectorize_cond_true_28607
3tag_vectorize_cond_pad_paddings_1_tag_vectorize_subL
Htag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
#tag_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2%
#tag_vectorize/cond/Pad/paddings/1/0�
!tag_vectorize/cond/Pad/paddings/1Pack,tag_vectorize/cond/Pad/paddings/1/0:output:03tag_vectorize_cond_pad_paddings_1_tag_vectorize_sub*
N*
T0*
_output_shapes
:2#
!tag_vectorize/cond/Pad/paddings/1�
#tag_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#tag_vectorize/cond/Pad/paddings/0_1�
tag_vectorize/cond/Pad/paddingsPack,tag_vectorize/cond/Pad/paddings/0_1:output:0*tag_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2!
tag_vectorize/cond/Pad/paddings�
tag_vectorize/cond/PadPadHtag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor(tag_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Pad�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
S
7__inference_global_average_pooling1d_layer_call_fn_3675

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_24872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������	:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
A__inference_dense_1_layer_call_and_return_conditional_losses_2532

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
y
$__inference_dense_layer_call_fn_3706

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_25052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
)
__inference_<lambda>_3820
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
�
�
id_vectorize_cond_false_2343!
id_vectorize_cond_placeholderT
Pid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
%id_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%id_vectorize/cond/strided_slice/stack�
'id_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'id_vectorize/cond/strided_slice/stack_1�
'id_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'id_vectorize/cond/strided_slice/stack_2�
id_vectorize/cond/strided_sliceStridedSlicePid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor.id_vectorize/cond/strided_slice/stack:output:00id_vectorize/cond/strided_slice/stack_1:output:00id_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2!
id_vectorize/cond/strided_slice�
id_vectorize/cond/IdentityIdentity(id_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
__inference_save_fn_3775
checkpoint_keyY
Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	��Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2�
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2J
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1�
IdentityIdentityadd:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const�

Identity_1IdentityConst:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1�

Identity_2IdentityOstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:keys:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2�

Identity_3Identity	add_1:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1�

Identity_4IdentityConst_1:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4�

Identity_5IdentityQstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:values:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2�
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
�
�
tag_vectorize_cond_true_33477
3tag_vectorize_cond_pad_paddings_1_tag_vectorize_subL
Htag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
#tag_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2%
#tag_vectorize/cond/Pad/paddings/1/0�
!tag_vectorize/cond/Pad/paddings/1Pack,tag_vectorize/cond/Pad/paddings/1/0:output:03tag_vectorize_cond_pad_paddings_1_tag_vectorize_sub*
N*
T0*
_output_shapes
:2#
!tag_vectorize/cond/Pad/paddings/1�
#tag_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#tag_vectorize/cond/Pad/paddings/0_1�
tag_vectorize/cond/Pad/paddingsPack,tag_vectorize/cond/Pad/paddings/0_1:output:0*tag_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2!
tag_vectorize/cond/Pad/paddings�
tag_vectorize/cond/PadPadHtag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor(tag_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Pad�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�	
�
"__inference_signature_wrapper_3160
id
item_tag
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalliditem_tagunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__wrapped_model_22642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
'
_output_shapes
:���������

_user_specified_nameid:QM
'
_output_shapes
:���������
"
_user_specified_name
item_tag:

_output_shapes
: :

_output_shapes
: 
�
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3670

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������	:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
id_vectorize_cond_true_32725
1id_vectorize_cond_pad_paddings_1_id_vectorize_subJ
Fid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
"id_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2$
"id_vectorize/cond/Pad/paddings/1/0�
 id_vectorize/cond/Pad/paddings/1Pack+id_vectorize/cond/Pad/paddings/1/0:output:01id_vectorize_cond_pad_paddings_1_id_vectorize_sub*
N*
T0*
_output_shapes
:2"
 id_vectorize/cond/Pad/paddings/1�
"id_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"id_vectorize/cond/Pad/paddings/0_1�
id_vectorize/cond/Pad/paddingsPack+id_vectorize/cond/Pad/paddings/0_1:output:0)id_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2 
id_vectorize/cond/Pad/paddings�
id_vectorize/cond/PadPadFid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor'id_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Pad�
id_vectorize/cond/IdentityIdentityid_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
?__inference_model_layer_call_and_return_conditional_losses_3101

inputs
inputs_1V
Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleW
Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	U
Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleV
Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
tag_embedding_layer_3081
id_embedding_layer_3084

dense_3090

dense_3092
dense_1_3095
dense_1_3097
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�*id_embedding_layer/StatefulPartitionedCall�Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2�+tag_embedding_layer/StatefulPartitionedCall�Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2t
id_vectorize/StringLowerStringLowerinputs*'
_output_shapes
:���������2
id_vectorize/StringLower�
id_vectorize/StaticRegexReplaceStaticRegexReplace!id_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2!
id_vectorize/StaticRegexReplace�
id_vectorize/SqueezeSqueeze(id_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
id_vectorize/Squeeze�
id_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2 
id_vectorize/StringSplit/Const�
&id_vectorize/StringSplit/StringSplitV2StringSplitV2id_vectorize/Squeeze:output:0'id_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2(
&id_vectorize/StringSplit/StringSplitV2�
,id_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,id_vectorize/StringSplit/strided_slice/stack�
.id_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.id_vectorize/StringSplit/strided_slice/stack_1�
.id_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.id_vectorize/StringSplit/strided_slice/stack_2�
&id_vectorize/StringSplit/strided_sliceStridedSlice0id_vectorize/StringSplit/StringSplitV2:indices:05id_vectorize/StringSplit/strided_slice/stack:output:07id_vectorize/StringSplit/strided_slice/stack_1:output:07id_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2(
&id_vectorize/StringSplit/strided_slice�
.id_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.id_vectorize/StringSplit/strided_slice_1/stack�
0id_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_1�
0id_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_2�
(id_vectorize/StringSplit/strided_slice_1StridedSlice.id_vectorize/StringSplit/StringSplitV2:shape:07id_vectorize/StringSplit/strided_slice_1/stack:output:09id_vectorize/StringSplit/strided_slice_1/stack_1:output:09id_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2*
(id_vectorize/StringSplit/strided_slice_1�
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast/id_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2Q
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast1id_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdbid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2_
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateraid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0fid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2`id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2^
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumcid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2\
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2cid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle/id_vectorize/StringSplit/StringSplitV2:values:0Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2G
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2~
.id_vectorize/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 20
.id_vectorize/string_lookup_1/assert_equal/NoOp�
%id_vectorize/string_lookup_1/IdentityIdentityNid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2'
%id_vectorize/string_lookup_1/Identity�
'id_vectorize/string_lookup_1/Identity_1IdentityZid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2)
'id_vectorize/string_lookup_1/Identity_1�
)id_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2+
)id_vectorize/RaggedToTensor/default_value�
!id_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2#
!id_vectorize/RaggedToTensor/Const�
0id_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor*id_vectorize/RaggedToTensor/Const:output:0.id_vectorize/string_lookup_1/Identity:output:02id_vectorize/RaggedToTensor/default_value:output:00id_vectorize/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS22
0id_vectorize/RaggedToTensor/RaggedTensorToTensor�
id_vectorize/ShapeShape9id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
id_vectorize/Shape�
 id_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 id_vectorize/strided_slice/stack�
"id_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_1�
"id_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_2�
id_vectorize/strided_sliceStridedSliceid_vectorize/Shape:output:0)id_vectorize/strided_slice/stack:output:0+id_vectorize/strided_slice/stack_1:output:0+id_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
id_vectorize/strided_slicej
id_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/sub/x�
id_vectorize/subSubid_vectorize/sub/x:output:0#id_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
id_vectorize/subl
id_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/Less/y�
id_vectorize/LessLess#id_vectorize/strided_slice:output:0id_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
id_vectorize/Less�
id_vectorize/condStatelessIfid_vectorize/Less:z:0id_vectorize/sub:z:09id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 */
else_branch R
id_vectorize_cond_false_2986*/
output_shapes
:������������������*.
then_branchR
id_vectorize_cond_true_29852
id_vectorize/cond�
id_vectorize/cond/IdentityIdentityid_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
id_vectorize/cond/Identityx
tag_vectorize/StringLowerStringLowerinputs_1*'
_output_shapes
:���������2
tag_vectorize/StringLower�
 tag_vectorize/StaticRegexReplaceStaticRegexReplace"tag_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2"
 tag_vectorize/StaticRegexReplace�
tag_vectorize/SqueezeSqueeze)tag_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
tag_vectorize/Squeeze�
tag_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2!
tag_vectorize/StringSplit/Const�
'tag_vectorize/StringSplit/StringSplitV2StringSplitV2tag_vectorize/Squeeze:output:0(tag_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2)
'tag_vectorize/StringSplit/StringSplitV2�
-tag_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tag_vectorize/StringSplit/strided_slice/stack�
/tag_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/tag_vectorize/StringSplit/strided_slice/stack_1�
/tag_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/tag_vectorize/StringSplit/strided_slice/stack_2�
'tag_vectorize/StringSplit/strided_sliceStridedSlice1tag_vectorize/StringSplit/StringSplitV2:indices:06tag_vectorize/StringSplit/strided_slice/stack:output:08tag_vectorize/StringSplit/strided_slice/stack_1:output:08tag_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'tag_vectorize/StringSplit/strided_slice�
/tag_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tag_vectorize/StringSplit/strided_slice_1/stack�
1tag_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_1�
1tag_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_2�
)tag_vectorize/StringSplit/strided_slice_1StridedSlice/tag_vectorize/StringSplit/StringSplitV2:shape:08tag_vectorize/StringSplit/strided_slice_1/stack:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_1:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2+
)tag_vectorize/StringSplit/strided_slice_1�
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast0tag_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2R
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast2tag_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2`
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterbtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0gtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2atag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2_
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumdtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2]
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2dtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle0tag_vectorize/StringSplit/StringSplitV2:values:0Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2F
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2|
-tag_vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2/
-tag_vectorize/string_lookup/assert_equal/NoOp�
$tag_vectorize/string_lookup/IdentityIdentityMtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2&
$tag_vectorize/string_lookup/Identity�
&tag_vectorize/string_lookup/Identity_1Identity[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2(
&tag_vectorize/string_lookup/Identity_1�
*tag_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2,
*tag_vectorize/RaggedToTensor/default_value�
"tag_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2$
"tag_vectorize/RaggedToTensor/Const�
1tag_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor+tag_vectorize/RaggedToTensor/Const:output:0-tag_vectorize/string_lookup/Identity:output:03tag_vectorize/RaggedToTensor/default_value:output:0/tag_vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS23
1tag_vectorize/RaggedToTensor/RaggedTensorToTensor�
tag_vectorize/ShapeShape:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
tag_vectorize/Shape�
!tag_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!tag_vectorize/strided_slice/stack�
#tag_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_1�
#tag_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_2�
tag_vectorize/strided_sliceStridedSlicetag_vectorize/Shape:output:0*tag_vectorize/strided_slice/stack:output:0,tag_vectorize/strided_slice/stack_1:output:0,tag_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
tag_vectorize/strided_slicel
tag_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/sub/x�
tag_vectorize/subSubtag_vectorize/sub/x:output:0$tag_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
tag_vectorize/subn
tag_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/Less/y�
tag_vectorize/LessLess$tag_vectorize/strided_slice:output:0tag_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
tag_vectorize/Less�
tag_vectorize/condStatelessIftag_vectorize/Less:z:0tag_vectorize/sub:z:0:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
else_branch!R
tag_vectorize_cond_false_3061*/
output_shapes
:������������������*/
then_branch R
tag_vectorize_cond_true_30602
tag_vectorize/cond�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
tag_vectorize/cond/Identity�
+tag_embedding_layer/StatefulPartitionedCallStatefulPartitionedCall$tag_vectorize/cond/Identity:output:0tag_embedding_layer_3081*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_tag_embedding_layer_layer_call_and_return_conditional_losses_24472-
+tag_embedding_layer/StatefulPartitionedCall�
*id_embedding_layer/StatefulPartitionedCallStatefulPartitionedCall#id_vectorize/cond/Identity:output:0id_embedding_layer_3084*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_id_embedding_layer_layer_call_and_return_conditional_losses_24682,
*id_embedding_layer/StatefulPartitionedCallp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis�
tf.concat/concatConcatV24tag_embedding_layer/StatefulPartitionedCall:output:03id_embedding_layer/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	2
tf.concat/concat�
(global_average_pooling1d/PartitionedCallPartitionedCalltf.concat/concat:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_24872*
(global_average_pooling1d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0
dense_3090
dense_3092*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_25052
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3095dense_1_3097*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_25322!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall+^id_embedding_layer/StatefulPartitionedCallF^id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2,^tag_embedding_layer/StatefulPartitionedCallE^tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*id_embedding_layer/StatefulPartitionedCall*id_embedding_layer/StatefulPartitionedCall2�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV22Z
+tag_embedding_layer/StatefulPartitionedCall+tag_embedding_layer/StatefulPartitionedCall2�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
��
�
?__inference_model_layer_call_and_return_conditional_losses_3580
	inputs_id
inputs_item_tagV
Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleW
Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	U
Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleV
Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	-
)tag_embedding_layer_embedding_lookup_3551,
(id_embedding_layer_embedding_lookup_3556(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�#id_embedding_layer/embedding_lookup�Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2�$tag_embedding_layer/embedding_lookup�Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2w
id_vectorize/StringLowerStringLower	inputs_id*'
_output_shapes
:���������2
id_vectorize/StringLower�
id_vectorize/StaticRegexReplaceStaticRegexReplace!id_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2!
id_vectorize/StaticRegexReplace�
id_vectorize/SqueezeSqueeze(id_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
id_vectorize/Squeeze�
id_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2 
id_vectorize/StringSplit/Const�
&id_vectorize/StringSplit/StringSplitV2StringSplitV2id_vectorize/Squeeze:output:0'id_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2(
&id_vectorize/StringSplit/StringSplitV2�
,id_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,id_vectorize/StringSplit/strided_slice/stack�
.id_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.id_vectorize/StringSplit/strided_slice/stack_1�
.id_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.id_vectorize/StringSplit/strided_slice/stack_2�
&id_vectorize/StringSplit/strided_sliceStridedSlice0id_vectorize/StringSplit/StringSplitV2:indices:05id_vectorize/StringSplit/strided_slice/stack:output:07id_vectorize/StringSplit/strided_slice/stack_1:output:07id_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2(
&id_vectorize/StringSplit/strided_slice�
.id_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.id_vectorize/StringSplit/strided_slice_1/stack�
0id_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_1�
0id_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_2�
(id_vectorize/StringSplit/strided_slice_1StridedSlice.id_vectorize/StringSplit/StringSplitV2:shape:07id_vectorize/StringSplit/strided_slice_1/stack:output:09id_vectorize/StringSplit/strided_slice_1/stack_1:output:09id_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2*
(id_vectorize/StringSplit/strided_slice_1�
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast/id_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2Q
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast1id_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdbid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2_
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateraid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0fid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2`id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2^
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumcid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2\
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2cid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle/id_vectorize/StringSplit/StringSplitV2:values:0Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2G
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2~
.id_vectorize/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 20
.id_vectorize/string_lookup_1/assert_equal/NoOp�
%id_vectorize/string_lookup_1/IdentityIdentityNid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2'
%id_vectorize/string_lookup_1/Identity�
'id_vectorize/string_lookup_1/Identity_1IdentityZid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2)
'id_vectorize/string_lookup_1/Identity_1�
)id_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2+
)id_vectorize/RaggedToTensor/default_value�
!id_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2#
!id_vectorize/RaggedToTensor/Const�
0id_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor*id_vectorize/RaggedToTensor/Const:output:0.id_vectorize/string_lookup_1/Identity:output:02id_vectorize/RaggedToTensor/default_value:output:00id_vectorize/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS22
0id_vectorize/RaggedToTensor/RaggedTensorToTensor�
id_vectorize/ShapeShape9id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
id_vectorize/Shape�
 id_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 id_vectorize/strided_slice/stack�
"id_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_1�
"id_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_2�
id_vectorize/strided_sliceStridedSliceid_vectorize/Shape:output:0)id_vectorize/strided_slice/stack:output:0+id_vectorize/strided_slice/stack_1:output:0+id_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
id_vectorize/strided_slicej
id_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/sub/x�
id_vectorize/subSubid_vectorize/sub/x:output:0#id_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
id_vectorize/subl
id_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/Less/y�
id_vectorize/LessLess#id_vectorize/strided_slice:output:0id_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
id_vectorize/Less�
id_vectorize/condStatelessIfid_vectorize/Less:z:0id_vectorize/sub:z:09id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 */
else_branch R
id_vectorize_cond_false_3456*/
output_shapes
:������������������*.
then_branchR
id_vectorize_cond_true_34552
id_vectorize/cond�
id_vectorize/cond/IdentityIdentityid_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
id_vectorize/cond/Identity
tag_vectorize/StringLowerStringLowerinputs_item_tag*'
_output_shapes
:���������2
tag_vectorize/StringLower�
 tag_vectorize/StaticRegexReplaceStaticRegexReplace"tag_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2"
 tag_vectorize/StaticRegexReplace�
tag_vectorize/SqueezeSqueeze)tag_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
tag_vectorize/Squeeze�
tag_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2!
tag_vectorize/StringSplit/Const�
'tag_vectorize/StringSplit/StringSplitV2StringSplitV2tag_vectorize/Squeeze:output:0(tag_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2)
'tag_vectorize/StringSplit/StringSplitV2�
-tag_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tag_vectorize/StringSplit/strided_slice/stack�
/tag_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/tag_vectorize/StringSplit/strided_slice/stack_1�
/tag_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/tag_vectorize/StringSplit/strided_slice/stack_2�
'tag_vectorize/StringSplit/strided_sliceStridedSlice1tag_vectorize/StringSplit/StringSplitV2:indices:06tag_vectorize/StringSplit/strided_slice/stack:output:08tag_vectorize/StringSplit/strided_slice/stack_1:output:08tag_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'tag_vectorize/StringSplit/strided_slice�
/tag_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tag_vectorize/StringSplit/strided_slice_1/stack�
1tag_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_1�
1tag_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_2�
)tag_vectorize/StringSplit/strided_slice_1StridedSlice/tag_vectorize/StringSplit/StringSplitV2:shape:08tag_vectorize/StringSplit/strided_slice_1/stack:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_1:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2+
)tag_vectorize/StringSplit/strided_slice_1�
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast0tag_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2R
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast2tag_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2`
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterbtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0gtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2atag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2_
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumdtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2]
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2dtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle0tag_vectorize/StringSplit/StringSplitV2:values:0Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2F
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2|
-tag_vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2/
-tag_vectorize/string_lookup/assert_equal/NoOp�
$tag_vectorize/string_lookup/IdentityIdentityMtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2&
$tag_vectorize/string_lookup/Identity�
&tag_vectorize/string_lookup/Identity_1Identity[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2(
&tag_vectorize/string_lookup/Identity_1�
*tag_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2,
*tag_vectorize/RaggedToTensor/default_value�
"tag_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2$
"tag_vectorize/RaggedToTensor/Const�
1tag_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor+tag_vectorize/RaggedToTensor/Const:output:0-tag_vectorize/string_lookup/Identity:output:03tag_vectorize/RaggedToTensor/default_value:output:0/tag_vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS23
1tag_vectorize/RaggedToTensor/RaggedTensorToTensor�
tag_vectorize/ShapeShape:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
tag_vectorize/Shape�
!tag_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!tag_vectorize/strided_slice/stack�
#tag_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_1�
#tag_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_2�
tag_vectorize/strided_sliceStridedSlicetag_vectorize/Shape:output:0*tag_vectorize/strided_slice/stack:output:0,tag_vectorize/strided_slice/stack_1:output:0,tag_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
tag_vectorize/strided_slicel
tag_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/sub/x�
tag_vectorize/subSubtag_vectorize/sub/x:output:0$tag_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
tag_vectorize/subn
tag_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/Less/y�
tag_vectorize/LessLess$tag_vectorize/strided_slice:output:0tag_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
tag_vectorize/Less�
tag_vectorize/condStatelessIftag_vectorize/Less:z:0tag_vectorize/sub:z:0:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
else_branch!R
tag_vectorize_cond_false_3531*/
output_shapes
:������������������*/
then_branch R
tag_vectorize_cond_true_35302
tag_vectorize/cond�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
tag_vectorize/cond/Identity�
$tag_embedding_layer/embedding_lookupResourceGather)tag_embedding_layer_embedding_lookup_3551$tag_vectorize/cond/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*<
_class2
0.loc:@tag_embedding_layer/embedding_lookup/3551*+
_output_shapes
:���������*
dtype02&
$tag_embedding_layer/embedding_lookup�
-tag_embedding_layer/embedding_lookup/IdentityIdentity-tag_embedding_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@tag_embedding_layer/embedding_lookup/3551*+
_output_shapes
:���������2/
-tag_embedding_layer/embedding_lookup/Identity�
/tag_embedding_layer/embedding_lookup/Identity_1Identity6tag_embedding_layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������21
/tag_embedding_layer/embedding_lookup/Identity_1�
#id_embedding_layer/embedding_lookupResourceGather(id_embedding_layer_embedding_lookup_3556#id_vectorize/cond/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*;
_class1
/-loc:@id_embedding_layer/embedding_lookup/3556*+
_output_shapes
:���������*
dtype02%
#id_embedding_layer/embedding_lookup�
,id_embedding_layer/embedding_lookup/IdentityIdentity,id_embedding_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@id_embedding_layer/embedding_lookup/3556*+
_output_shapes
:���������2.
,id_embedding_layer/embedding_lookup/Identity�
.id_embedding_layer/embedding_lookup/Identity_1Identity5id_embedding_layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������20
.id_embedding_layer/embedding_lookup/Identity_1p
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis�
tf.concat/concatConcatV28tag_embedding_layer/embedding_lookup/Identity_1:output:07id_embedding_layer/embedding_lookup/Identity_1:output:0tf.concat/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	2
tf.concat/concat�
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices�
global_average_pooling1d/MeanMeantf.concat/concat:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������2
global_average_pooling1d/Mean�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Sigmoid�
IdentityIdentitydense_1/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp$^id_embedding_layer/embedding_lookupF^id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2%^tag_embedding_layer/embedding_lookupE^tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2J
#id_embedding_layer/embedding_lookup#id_embedding_layer/embedding_lookup2�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV22L
$tag_embedding_layer/embedding_lookup$tag_embedding_layer/embedding_lookup2�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:R N
'
_output_shapes
:���������
#
_user_specified_name	inputs/id:XT
'
_output_shapes
:���������
)
_user_specified_nameinputs/item_tag:

_output_shapes
: :

_output_shapes
: 
�
�
id_vectorize_cond_false_3456!
id_vectorize_cond_placeholderT
Pid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
%id_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%id_vectorize/cond/strided_slice/stack�
'id_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'id_vectorize/cond/strided_slice/stack_1�
'id_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'id_vectorize/cond/strided_slice/stack_2�
id_vectorize/cond/strided_sliceStridedSlicePid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor.id_vectorize/cond/strided_slice/stack:output:00id_vectorize/cond/strided_slice/stack_1:output:00id_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2!
id_vectorize/cond/strided_slice�
id_vectorize/cond/IdentityIdentity(id_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�	
�
__inference_restore_fn_3810
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_1_index_table_table_restore_lookuptableimportv2_table_handle
identity��=string_lookup_1_index_table_table_restore/LookupTableImportV2�
=string_lookup_1_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_1_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const�
IdentityIdentityConst:output:0>^string_lookup_1_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:::2~
=string_lookup_1_index_table_table_restore/LookupTableImportV2=string_lookup_1_index_table_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
�
�
tag_vectorize_cond_true_24177
3tag_vectorize_cond_pad_paddings_1_tag_vectorize_subL
Htag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
#tag_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2%
#tag_vectorize/cond/Pad/paddings/1/0�
!tag_vectorize/cond/Pad/paddings/1Pack,tag_vectorize/cond/Pad/paddings/1/0:output:03tag_vectorize_cond_pad_paddings_1_tag_vectorize_sub*
N*
T0*
_output_shapes
:2#
!tag_vectorize/cond/Pad/paddings/1�
#tag_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#tag_vectorize/cond/Pad/paddings/0_1�
tag_vectorize/cond/Pad/paddingsPack,tag_vectorize/cond/Pad/paddings/0_1:output:0*tag_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2!
tag_vectorize/cond/Pad/paddings�
tag_vectorize/cond/PadPadHtag_vectorize_cond_pad_tag_vectorize_raggedtotensor_raggedtensortotensor(tag_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Pad�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
+
__inference__destroyer_3741
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
�
�
id_vectorize_cond_true_34555
1id_vectorize_cond_pad_paddings_1_id_vectorize_subJ
Fid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
"id_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2$
"id_vectorize/cond/Pad/paddings/1/0�
 id_vectorize/cond/Pad/paddings/1Pack+id_vectorize/cond/Pad/paddings/1/0:output:01id_vectorize_cond_pad_paddings_1_id_vectorize_sub*
N*
T0*
_output_shapes
:2"
 id_vectorize/cond/Pad/paddings/1�
"id_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"id_vectorize/cond/Pad/paddings/0_1�
id_vectorize/cond/Pad/paddingsPack+id_vectorize/cond/Pad/paddings/0_1:output:0)id_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2 
id_vectorize/cond/Pad/paddings�
id_vectorize/cond/PadPadFid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor'id_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Pad�
id_vectorize/cond/IdentityIdentityid_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
J
__inference__creator_3746
identity��string_lookup_1_index_table�
string_lookup_1_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_9*
value_dtype0	2
string_lookup_1_index_table�
IdentityIdentity*string_lookup_1_index_table:table_handle:0^string_lookup_1_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2:
string_lookup_1_index_tablestring_lookup_1_index_table
�	
�
?__inference_dense_layer_call_and_return_conditional_losses_2505

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3681

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
tag_vectorize_cond_false_3348"
tag_vectorize_cond_placeholderV
Rtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
&tag_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&tag_vectorize/cond/strided_slice/stack�
(tag_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(tag_vectorize/cond/strided_slice/stack_1�
(tag_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(tag_vectorize/cond/strided_slice/stack_2�
 tag_vectorize/cond/strided_sliceStridedSliceRtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor/tag_vectorize/cond/strided_slice/stack:output:01tag_vectorize/cond/strided_slice/stack_1:output:01tag_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2"
 tag_vectorize/cond/strided_slice�
tag_vectorize/cond/IdentityIdentity)tag_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
id_vectorize_cond_true_26075
1id_vectorize_cond_pad_paddings_1_id_vectorize_subJ
Fid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
"id_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2$
"id_vectorize/cond/Pad/paddings/1/0�
 id_vectorize/cond/Pad/paddings/1Pack+id_vectorize/cond/Pad/paddings/1/0:output:01id_vectorize_cond_pad_paddings_1_id_vectorize_sub*
N*
T0*
_output_shapes
:2"
 id_vectorize/cond/Pad/paddings/1�
"id_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"id_vectorize/cond/Pad/paddings/0_1�
id_vectorize/cond/Pad/paddingsPack+id_vectorize/cond/Pad/paddings/0_1:output:0)id_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2 
id_vectorize/cond/Pad/paddings�
id_vectorize/cond/PadPadFid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor'id_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Pad�
id_vectorize/cond/IdentityIdentityid_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
#model_tag_vectorize_cond_false_2215(
$model_tag_vectorize_cond_placeholderb
^model_tag_vectorize_cond_strided_slice_model_tag_vectorize_raggedtotensor_raggedtensortotensor	%
!model_tag_vectorize_cond_identity	�
,model/tag_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,model/tag_vectorize/cond/strided_slice/stack�
.model/tag_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.model/tag_vectorize/cond/strided_slice/stack_1�
.model/tag_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.model/tag_vectorize/cond/strided_slice/stack_2�
&model/tag_vectorize/cond/strided_sliceStridedSlice^model_tag_vectorize_cond_strided_slice_model_tag_vectorize_raggedtotensor_raggedtensortotensor5model/tag_vectorize/cond/strided_slice/stack:output:07model/tag_vectorize/cond/strided_slice/stack_1:output:07model/tag_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2(
&model/tag_vectorize/cond/strided_slice�
!model/tag_vectorize/cond/IdentityIdentity/model/tag_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2#
!model/tag_vectorize/cond/Identity"O
!model_tag_vectorize_cond_identity*model/tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
L__inference_id_embedding_layer_layer_call_and_return_conditional_losses_2468

inputs	
embedding_lookup_2462
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_2462inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/2462*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/2462*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
id_vectorize_cond_true_27855
1id_vectorize_cond_pad_paddings_1_id_vectorize_subJ
Fid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
"id_vectorize/cond/Pad/paddings/1/0Const*
_output_shapes
: *
dtype0*
value	B : 2$
"id_vectorize/cond/Pad/paddings/1/0�
 id_vectorize/cond/Pad/paddings/1Pack+id_vectorize/cond/Pad/paddings/1/0:output:01id_vectorize_cond_pad_paddings_1_id_vectorize_sub*
N*
T0*
_output_shapes
:2"
 id_vectorize/cond/Pad/paddings/1�
"id_vectorize/cond/Pad/paddings/0_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"id_vectorize/cond/Pad/paddings/0_1�
id_vectorize/cond/Pad/paddingsPack+id_vectorize/cond/Pad/paddings/0_1:output:0)id_vectorize/cond/Pad/paddings/1:output:0*
N*
T0*
_output_shapes

:2 
id_vectorize/cond/Pad/paddings�
id_vectorize/cond/PadPadFid_vectorize_cond_pad_id_vectorize_raggedtotensor_raggedtensortotensor'id_vectorize/cond/Pad/paddings:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Pad�
id_vectorize/cond/IdentityIdentityid_vectorize/cond/Pad:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
id_vectorize_cond_false_2786!
id_vectorize_cond_placeholderT
Pid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
%id_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%id_vectorize/cond/strided_slice/stack�
'id_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'id_vectorize/cond/strided_slice/stack_1�
'id_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'id_vectorize/cond/strided_slice/stack_2�
id_vectorize/cond/strided_sliceStridedSlicePid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor.id_vectorize/cond/strided_slice/stack:output:00id_vectorize/cond/strided_slice/stack_1:output:00id_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2!
id_vectorize/cond/strided_slice�
id_vectorize/cond/IdentityIdentity(id_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�
�
id_vectorize_cond_false_2986!
id_vectorize_cond_placeholderT
Pid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor	
id_vectorize_cond_identity	�
%id_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%id_vectorize/cond/strided_slice/stack�
'id_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'id_vectorize/cond/strided_slice/stack_1�
'id_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'id_vectorize/cond/strided_slice/stack_2�
id_vectorize/cond/strided_sliceStridedSlicePid_vectorize_cond_strided_slice_id_vectorize_raggedtotensor_raggedtensortotensor.id_vectorize/cond/strided_slice/stack:output:00id_vectorize/cond/strided_slice/stack_1:output:00id_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2!
id_vectorize/cond/strided_slice�
id_vectorize/cond/IdentityIdentity(id_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
id_vectorize/cond/Identity"A
id_vectorize_cond_identity#id_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
��
�
 __inference__traced_restore_4048
file_prefix3
/assignvariableop_tag_embedding_layer_embeddings4
0assignvariableop_1_id_embedding_layer_embeddings#
assignvariableop_2_dense_kernel!
assignvariableop_3_dense_bias%
!assignvariableop_4_dense_1_kernel#
assignvariableop_5_dense_1_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rateY
Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_table]
Ystring_lookup_1_index_table_table_restore_lookuptableimportv2_string_lookup_1_index_table
assignvariableop_11_total
assignvariableop_12_count&
"assignvariableop_13_true_positives&
"assignvariableop_14_true_negatives'
#assignvariableop_15_false_positives'
#assignvariableop_16_false_negatives=
9assignvariableop_17_adam_tag_embedding_layer_embeddings_m<
8assignvariableop_18_adam_id_embedding_layer_embeddings_m+
'assignvariableop_19_adam_dense_kernel_m)
%assignvariableop_20_adam_dense_bias_m-
)assignvariableop_21_adam_dense_1_kernel_m+
'assignvariableop_22_adam_dense_1_bias_m=
9assignvariableop_23_adam_tag_embedding_layer_embeddings_v<
8assignvariableop_24_adam_id_embedding_layer_embeddings_v+
'assignvariableop_25_adam_dense_kernel_v)
%assignvariableop_26_adam_dense_bias_v-
)assignvariableop_27_adam_dense_1_kernel_v+
'assignvariableop_28_adam_dense_1_bias_v
identity_30��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�=string_lookup_1_index_table_table_restore/LookupTableImportV2�;string_lookup_index_table_table_restore/LookupTableImportV2�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*�
value�B�"B:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesBFlayer_with_weights-1/_index_lookup_layer/_table/.ATTRIBUTES/table-keysBHlayer_with_weights-1/_index_lookup_layer/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::*0
dtypes&
$2"			2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp/assignvariableop_tag_embedding_layer_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp0assignvariableop_1_id_embedding_layer_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10�
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_tableRestoreV2:tensors:11RestoreV2:tensors:12*	
Tin0*

Tout0	*,
_class"
 loc:@string_lookup_index_table*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2�
=string_lookup_1_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_1_index_table_table_restore_lookuptableimportv2_string_lookup_1_index_tableRestoreV2:tensors:13RestoreV2:tensors:14*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_1_index_table*
_output_shapes
 2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2n
Identity_11IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_true_positivesIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_true_negativesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_false_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_adam_tag_embedding_layer_embeddings_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adam_id_embedding_layer_embeddings_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_adam_tag_embedding_layer_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp8assignvariableop_24_adam_id_embedding_layer_embeddings_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp>^string_lookup_1_index_table_table_restore/LookupTableImportV2<^string_lookup_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29�
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9>^string_lookup_1_index_table_table_restore/LookupTableImportV2<^string_lookup_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*�
_input_shapes�
~: :::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92~
=string_lookup_1_index_table_table_restore/LookupTableImportV2=string_lookup_1_index_table_table_restore/LookupTableImportV22z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:2.
,
_class"
 loc:@string_lookup_index_table:40
.
_class$
" loc:@string_lookup_1_index_table
�
�
__inference__wrapped_model_2264
id
item_tag\
Xmodel_id_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle]
Ymodel_id_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	[
Wmodel_tag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle\
Xmodel_tag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	3
/model_tag_embedding_layer_embedding_lookup_22352
.model_id_embedding_layer_embedding_lookup_2240.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�)model/id_embedding_layer/embedding_lookup�Kmodel/id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2�*model/tag_embedding_layer/embedding_lookup�Jmodel/tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2|
model/id_vectorize/StringLowerStringLowerid*'
_output_shapes
:���������2 
model/id_vectorize/StringLower�
%model/id_vectorize/StaticRegexReplaceStaticRegexReplace'model/id_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2'
%model/id_vectorize/StaticRegexReplace�
model/id_vectorize/SqueezeSqueeze.model/id_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
model/id_vectorize/Squeeze�
$model/id_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2&
$model/id_vectorize/StringSplit/Const�
,model/id_vectorize/StringSplit/StringSplitV2StringSplitV2#model/id_vectorize/Squeeze:output:0-model/id_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2.
,model/id_vectorize/StringSplit/StringSplitV2�
2model/id_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2model/id_vectorize/StringSplit/strided_slice/stack�
4model/id_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model/id_vectorize/StringSplit/strided_slice/stack_1�
4model/id_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4model/id_vectorize/StringSplit/strided_slice/stack_2�
,model/id_vectorize/StringSplit/strided_sliceStridedSlice6model/id_vectorize/StringSplit/StringSplitV2:indices:0;model/id_vectorize/StringSplit/strided_slice/stack:output:0=model/id_vectorize/StringSplit/strided_slice/stack_1:output:0=model/id_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2.
,model/id_vectorize/StringSplit/strided_slice�
4model/id_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4model/id_vectorize/StringSplit/strided_slice_1/stack�
6model/id_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6model/id_vectorize/StringSplit/strided_slice_1/stack_1�
6model/id_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6model/id_vectorize/StringSplit/strided_slice_1/stack_2�
.model/id_vectorize/StringSplit/strided_slice_1StridedSlice4model/id_vectorize/StringSplit/StringSplitV2:shape:0=model/id_vectorize/StringSplit/strided_slice_1/stack:output:0?model/id_vectorize/StringSplit/strided_slice_1/stack_1:output:0?model/id_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask20
.model/id_vectorize/StringSplit/strided_slice_1�
Umodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5model/id_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2W
Umodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Wmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7model/id_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2Y
Wmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
_model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2a
_model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
_model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2a
_model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
^model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2`
^model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
cmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2e
cmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2c
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
^model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastemodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2`
^model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2c
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
]model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2_
]model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
_model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2a
_model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
]model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2_
]model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
]model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2_
]model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2c
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0emodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2c
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2c
amodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
bmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0emodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2d
bmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
\model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Wmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumimodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0emodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2Y
Wmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
`model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2b
`model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
\model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Wmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2imodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0emodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2Y
Wmodel/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Kmodel/id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Xmodel_id_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle5model/id_vectorize/StringSplit/StringSplitV2:values:0Ymodel_id_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2M
Kmodel/id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2�
4model/id_vectorize/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 26
4model/id_vectorize/string_lookup_1/assert_equal/NoOp�
+model/id_vectorize/string_lookup_1/IdentityIdentityTmodel/id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2-
+model/id_vectorize/string_lookup_1/Identity�
-model/id_vectorize/string_lookup_1/Identity_1Identity`model/id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2/
-model/id_vectorize/string_lookup_1/Identity_1�
/model/id_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 21
/model/id_vectorize/RaggedToTensor/default_value�
'model/id_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2)
'model/id_vectorize/RaggedToTensor/Const�
6model/id_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0model/id_vectorize/RaggedToTensor/Const:output:04model/id_vectorize/string_lookup_1/Identity:output:08model/id_vectorize/RaggedToTensor/default_value:output:06model/id_vectorize/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS28
6model/id_vectorize/RaggedToTensor/RaggedTensorToTensor�
model/id_vectorize/ShapeShape?model/id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
model/id_vectorize/Shape�
&model/id_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&model/id_vectorize/strided_slice/stack�
(model/id_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model/id_vectorize/strided_slice/stack_1�
(model/id_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model/id_vectorize/strided_slice/stack_2�
 model/id_vectorize/strided_sliceStridedSlice!model/id_vectorize/Shape:output:0/model/id_vectorize/strided_slice/stack:output:01model/id_vectorize/strided_slice/stack_1:output:01model/id_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model/id_vectorize/strided_slicev
model/id_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
model/id_vectorize/sub/x�
model/id_vectorize/subSub!model/id_vectorize/sub/x:output:0)model/id_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
model/id_vectorize/subx
model/id_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/id_vectorize/Less/y�
model/id_vectorize/LessLess)model/id_vectorize/strided_slice:output:0"model/id_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
model/id_vectorize/Less�
model/id_vectorize/condStatelessIfmodel/id_vectorize/Less:z:0model/id_vectorize/sub:z:0?model/id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *5
else_branch&R$
"model_id_vectorize_cond_false_2140*/
output_shapes
:������������������*4
then_branch%R#
!model_id_vectorize_cond_true_21392
model/id_vectorize/cond�
 model/id_vectorize/cond/IdentityIdentity model/id_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2"
 model/id_vectorize/cond/Identity�
model/tag_vectorize/StringLowerStringLoweritem_tag*'
_output_shapes
:���������2!
model/tag_vectorize/StringLower�
&model/tag_vectorize/StaticRegexReplaceStaticRegexReplace(model/tag_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2(
&model/tag_vectorize/StaticRegexReplace�
model/tag_vectorize/SqueezeSqueeze/model/tag_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
model/tag_vectorize/Squeeze�
%model/tag_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2'
%model/tag_vectorize/StringSplit/Const�
-model/tag_vectorize/StringSplit/StringSplitV2StringSplitV2$model/tag_vectorize/Squeeze:output:0.model/tag_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2/
-model/tag_vectorize/StringSplit/StringSplitV2�
3model/tag_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3model/tag_vectorize/StringSplit/strided_slice/stack�
5model/tag_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       27
5model/tag_vectorize/StringSplit/strided_slice/stack_1�
5model/tag_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5model/tag_vectorize/StringSplit/strided_slice/stack_2�
-model/tag_vectorize/StringSplit/strided_sliceStridedSlice7model/tag_vectorize/StringSplit/StringSplitV2:indices:0<model/tag_vectorize/StringSplit/strided_slice/stack:output:0>model/tag_vectorize/StringSplit/strided_slice/stack_1:output:0>model/tag_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2/
-model/tag_vectorize/StringSplit/strided_slice�
5model/tag_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5model/tag_vectorize/StringSplit/strided_slice_1/stack�
7model/tag_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model/tag_vectorize/StringSplit/strided_slice_1/stack_1�
7model/tag_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model/tag_vectorize/StringSplit/strided_slice_1/stack_2�
/model/tag_vectorize/StringSplit/strided_slice_1StridedSlice5model/tag_vectorize/StringSplit/StringSplitV2:shape:0>model/tag_vectorize/StringSplit/strided_slice_1/stack:output:0@model/tag_vectorize/StringSplit/strided_slice_1/stack_1:output:0@model/tag_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask21
/model/tag_vectorize/StringSplit/strided_slice_1�
Vmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast6model/tag_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2X
Vmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Xmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast8model/tag_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2Z
Xmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
`model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeZmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2b
`model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
`model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2b
`model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
_model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdimodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0imodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2a
_model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
dmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2f
dmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterhmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0mmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2d
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
_model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastfmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2a
_model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2d
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
^model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxZmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0kmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2`
^model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
`model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2b
`model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
^model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2gmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0imodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2`
^model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
^model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulcmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2`
^model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum\model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2d
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum\model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0fmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2d
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2d
bmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
cmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountZmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0fmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0kmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2e
cmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
]model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2_
]model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Xmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumjmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0fmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2Z
Xmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
amodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2c
amodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
]model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2_
]model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Xmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2jmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0^model/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0fmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2Z
Xmodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Jmodel/tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Wmodel_tag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle6model/tag_vectorize/StringSplit/StringSplitV2:values:0Xmodel_tag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2L
Jmodel/tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2�
3model/tag_vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 25
3model/tag_vectorize/string_lookup/assert_equal/NoOp�
*model/tag_vectorize/string_lookup/IdentityIdentitySmodel/tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2,
*model/tag_vectorize/string_lookup/Identity�
,model/tag_vectorize/string_lookup/Identity_1Identityamodel/tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2.
,model/tag_vectorize/string_lookup/Identity_1�
0model/tag_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 22
0model/tag_vectorize/RaggedToTensor/default_value�
(model/tag_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2*
(model/tag_vectorize/RaggedToTensor/Const�
7model/tag_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor1model/tag_vectorize/RaggedToTensor/Const:output:03model/tag_vectorize/string_lookup/Identity:output:09model/tag_vectorize/RaggedToTensor/default_value:output:05model/tag_vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS29
7model/tag_vectorize/RaggedToTensor/RaggedTensorToTensor�
model/tag_vectorize/ShapeShape@model/tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
model/tag_vectorize/Shape�
'model/tag_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'model/tag_vectorize/strided_slice/stack�
)model/tag_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model/tag_vectorize/strided_slice/stack_1�
)model/tag_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model/tag_vectorize/strided_slice/stack_2�
!model/tag_vectorize/strided_sliceStridedSlice"model/tag_vectorize/Shape:output:00model/tag_vectorize/strided_slice/stack:output:02model/tag_vectorize/strided_slice/stack_1:output:02model/tag_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model/tag_vectorize/strided_slicex
model/tag_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
model/tag_vectorize/sub/x�
model/tag_vectorize/subSub"model/tag_vectorize/sub/x:output:0*model/tag_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
model/tag_vectorize/subz
model/tag_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/tag_vectorize/Less/y�
model/tag_vectorize/LessLess*model/tag_vectorize/strided_slice:output:0#model/tag_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
model/tag_vectorize/Less�
model/tag_vectorize/condStatelessIfmodel/tag_vectorize/Less:z:0model/tag_vectorize/sub:z:0@model/tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *6
else_branch'R%
#model_tag_vectorize_cond_false_2215*/
output_shapes
:������������������*5
then_branch&R$
"model_tag_vectorize_cond_true_22142
model/tag_vectorize/cond�
!model/tag_vectorize/cond/IdentityIdentity!model/tag_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2#
!model/tag_vectorize/cond/Identity�
*model/tag_embedding_layer/embedding_lookupResourceGather/model_tag_embedding_layer_embedding_lookup_2235*model/tag_vectorize/cond/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*B
_class8
64loc:@model/tag_embedding_layer/embedding_lookup/2235*+
_output_shapes
:���������*
dtype02,
*model/tag_embedding_layer/embedding_lookup�
3model/tag_embedding_layer/embedding_lookup/IdentityIdentity3model/tag_embedding_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model/tag_embedding_layer/embedding_lookup/2235*+
_output_shapes
:���������25
3model/tag_embedding_layer/embedding_lookup/Identity�
5model/tag_embedding_layer/embedding_lookup/Identity_1Identity<model/tag_embedding_layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������27
5model/tag_embedding_layer/embedding_lookup/Identity_1�
)model/id_embedding_layer/embedding_lookupResourceGather.model_id_embedding_layer_embedding_lookup_2240)model/id_vectorize/cond/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@model/id_embedding_layer/embedding_lookup/2240*+
_output_shapes
:���������*
dtype02+
)model/id_embedding_layer/embedding_lookup�
2model/id_embedding_layer/embedding_lookup/IdentityIdentity2model/id_embedding_layer/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model/id_embedding_layer/embedding_lookup/2240*+
_output_shapes
:���������24
2model/id_embedding_layer/embedding_lookup/Identity�
4model/id_embedding_layer/embedding_lookup/Identity_1Identity;model/id_embedding_layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������26
4model/id_embedding_layer/embedding_lookup/Identity_1|
model/tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/tf.concat/concat/axis�
model/tf.concat/concatConcatV2>model/tag_embedding_layer/embedding_lookup/Identity_1:output:0=model/id_embedding_layer/embedding_lookup/Identity_1:output:0$model/tf.concat/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	2
model/tf.concat/concat�
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :27
5model/global_average_pooling1d/Mean/reduction_indices�
#model/global_average_pooling1d/MeanMeanmodel/tf.concat/concat:output:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������2%
#model/global_average_pooling1d/Mean�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMul,model/global_average_pooling1d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/dense/Relu�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense_1/MatMul/ReadVariableOp�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_1/MatMul�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_1/BiasAdd�
model/dense_1/SigmoidSigmoidmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/dense_1/Sigmoid�
IdentityIdentitymodel/dense_1/Sigmoid:y:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp*^model/id_embedding_layer/embedding_lookupL^model/id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2+^model/tag_embedding_layer/embedding_lookupK^model/tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2V
)model/id_embedding_layer/embedding_lookup)model/id_embedding_layer/embedding_lookup2�
Kmodel/id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2Kmodel/id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV22X
*model/tag_embedding_layer/embedding_lookup*model/tag_embedding_layer/embedding_lookup2�
Jmodel/tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2Jmodel/tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:K G
'
_output_shapes
:���������

_user_specified_nameid:QM
'
_output_shapes
:���������
"
_user_specified_name
item_tag:

_output_shapes
: :

_output_shapes
: 
ۆ
�
?__inference_model_layer_call_and_return_conditional_losses_2549
id
item_tagV
Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleW
Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	U
Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleV
Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	
tag_embedding_layer_2456
id_embedding_layer_2477

dense_2516

dense_2518
dense_1_2543
dense_1_2545
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�*id_embedding_layer/StatefulPartitionedCall�Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2�+tag_embedding_layer/StatefulPartitionedCall�Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2p
id_vectorize/StringLowerStringLowerid*'
_output_shapes
:���������2
id_vectorize/StringLower�
id_vectorize/StaticRegexReplaceStaticRegexReplace!id_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2!
id_vectorize/StaticRegexReplace�
id_vectorize/SqueezeSqueeze(id_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
id_vectorize/Squeeze�
id_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2 
id_vectorize/StringSplit/Const�
&id_vectorize/StringSplit/StringSplitV2StringSplitV2id_vectorize/Squeeze:output:0'id_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2(
&id_vectorize/StringSplit/StringSplitV2�
,id_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,id_vectorize/StringSplit/strided_slice/stack�
.id_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.id_vectorize/StringSplit/strided_slice/stack_1�
.id_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.id_vectorize/StringSplit/strided_slice/stack_2�
&id_vectorize/StringSplit/strided_sliceStridedSlice0id_vectorize/StringSplit/StringSplitV2:indices:05id_vectorize/StringSplit/strided_slice/stack:output:07id_vectorize/StringSplit/strided_slice/stack_1:output:07id_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2(
&id_vectorize/StringSplit/strided_slice�
.id_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.id_vectorize/StringSplit/strided_slice_1/stack�
0id_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_1�
0id_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0id_vectorize/StringSplit/strided_slice_1/stack_2�
(id_vectorize/StringSplit/strided_slice_1StridedSlice.id_vectorize/StringSplit/StringSplitV2:shape:07id_vectorize/StringSplit/strided_slice_1/stack:output:09id_vectorize/StringSplit/strided_slice_1/stack_1:output:09id_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2*
(id_vectorize/StringSplit/strided_slice_1�
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast/id_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2Q
Oid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast1id_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdbid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2_
]id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateraid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0fid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2Z
Xid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2[
Yid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2`id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0bid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Y
Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumUid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2]
[id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountSid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0did_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2^
\id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumcid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2\
Zid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2cid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Wid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0_id_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2S
Qid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle/id_vectorize/StringSplit/StringSplitV2:values:0Sid_vectorize_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2G
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2~
.id_vectorize/string_lookup_1/assert_equal/NoOpNoOp*
_output_shapes
 20
.id_vectorize/string_lookup_1/assert_equal/NoOp�
%id_vectorize/string_lookup_1/IdentityIdentityNid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2'
%id_vectorize/string_lookup_1/Identity�
'id_vectorize/string_lookup_1/Identity_1IdentityZid_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2)
'id_vectorize/string_lookup_1/Identity_1�
)id_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2+
)id_vectorize/RaggedToTensor/default_value�
!id_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2#
!id_vectorize/RaggedToTensor/Const�
0id_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor*id_vectorize/RaggedToTensor/Const:output:0.id_vectorize/string_lookup_1/Identity:output:02id_vectorize/RaggedToTensor/default_value:output:00id_vectorize/string_lookup_1/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS22
0id_vectorize/RaggedToTensor/RaggedTensorToTensor�
id_vectorize/ShapeShape9id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
id_vectorize/Shape�
 id_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 id_vectorize/strided_slice/stack�
"id_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_1�
"id_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"id_vectorize/strided_slice/stack_2�
id_vectorize/strided_sliceStridedSliceid_vectorize/Shape:output:0)id_vectorize/strided_slice/stack:output:0+id_vectorize/strided_slice/stack_1:output:0+id_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
id_vectorize/strided_slicej
id_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/sub/x�
id_vectorize/subSubid_vectorize/sub/x:output:0#id_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
id_vectorize/subl
id_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
id_vectorize/Less/y�
id_vectorize/LessLess#id_vectorize/strided_slice:output:0id_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
id_vectorize/Less�
id_vectorize/condStatelessIfid_vectorize/Less:z:0id_vectorize/sub:z:09id_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 */
else_branch R
id_vectorize_cond_false_2343*/
output_shapes
:������������������*.
then_branchR
id_vectorize_cond_true_23422
id_vectorize/cond�
id_vectorize/cond/IdentityIdentityid_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
id_vectorize/cond/Identityx
tag_vectorize/StringLowerStringLoweritem_tag*'
_output_shapes
:���������2
tag_vectorize/StringLower�
 tag_vectorize/StaticRegexReplaceStaticRegexReplace"tag_vectorize/StringLower:output:0*'
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite 2"
 tag_vectorize/StaticRegexReplace�
tag_vectorize/SqueezeSqueeze)tag_vectorize/StaticRegexReplace:output:0*
T0*#
_output_shapes
:���������*
squeeze_dims

���������2
tag_vectorize/Squeeze�
tag_vectorize/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B 2!
tag_vectorize/StringSplit/Const�
'tag_vectorize/StringSplit/StringSplitV2StringSplitV2tag_vectorize/Squeeze:output:0(tag_vectorize/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:2)
'tag_vectorize/StringSplit/StringSplitV2�
-tag_vectorize/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-tag_vectorize/StringSplit/strided_slice/stack�
/tag_vectorize/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/tag_vectorize/StringSplit/strided_slice/stack_1�
/tag_vectorize/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/tag_vectorize/StringSplit/strided_slice/stack_2�
'tag_vectorize/StringSplit/strided_sliceStridedSlice1tag_vectorize/StringSplit/StringSplitV2:indices:06tag_vectorize/StringSplit/strided_slice/stack:output:08tag_vectorize/StringSplit/strided_slice/stack_1:output:08tag_vectorize/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'tag_vectorize/StringSplit/strided_slice�
/tag_vectorize/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tag_vectorize/StringSplit/strided_slice_1/stack�
1tag_vectorize/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_1�
1tag_vectorize/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tag_vectorize/StringSplit/strided_slice_1/stack_2�
)tag_vectorize/StringSplit/strided_slice_1StridedSlice/tag_vectorize/StringSplit/StringSplitV2:shape:08tag_vectorize/StringSplit/strided_slice_1/stack:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_1:output:0:tag_vectorize/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask2+
)tag_vectorize/StringSplit/strided_slice_1�
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast0tag_vectorize/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2R
Ptag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast2tag_vectorize/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod�
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2`
^tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterbtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0gtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater�
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: 2[
Ytag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max�
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :2\
Ztag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2atag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ctag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add�
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: 2Z
Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumVtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum�
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 2^
\tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2�
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountTtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0etag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:���������2_
]tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumdtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum�
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R 2]
[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0�
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Y
Wtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis�
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2dtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0Xtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0`tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:���������2T
Rtag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Qtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle0tag_vectorize/StringSplit/StringSplitV2:values:0Rtag_vectorize_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*#
_output_shapes
:���������2F
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2|
-tag_vectorize/string_lookup/assert_equal/NoOpNoOp*
_output_shapes
 2/
-tag_vectorize/string_lookup/assert_equal/NoOp�
$tag_vectorize/string_lookup/IdentityIdentityMtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:���������2&
$tag_vectorize/string_lookup/Identity�
&tag_vectorize/string_lookup/Identity_1Identity[tag_vectorize/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*
T0	*#
_output_shapes
:���������2(
&tag_vectorize/string_lookup/Identity_1�
*tag_vectorize/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 2,
*tag_vectorize/RaggedToTensor/default_value�
"tag_vectorize/RaggedToTensor/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2$
"tag_vectorize/RaggedToTensor/Const�
1tag_vectorize/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor+tag_vectorize/RaggedToTensor/Const:output:0-tag_vectorize/string_lookup/Identity:output:03tag_vectorize/RaggedToTensor/default_value:output:0/tag_vectorize/string_lookup/Identity_1:output:0*
T0	*
Tindex0	*
Tshape0	*0
_output_shapes
:������������������*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS23
1tag_vectorize/RaggedToTensor/RaggedTensorToTensor�
tag_vectorize/ShapeShape:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
T0	*
_output_shapes
:2
tag_vectorize/Shape�
!tag_vectorize/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!tag_vectorize/strided_slice/stack�
#tag_vectorize/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_1�
#tag_vectorize/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#tag_vectorize/strided_slice/stack_2�
tag_vectorize/strided_sliceStridedSlicetag_vectorize/Shape:output:0*tag_vectorize/strided_slice/stack:output:0,tag_vectorize/strided_slice/stack_1:output:0,tag_vectorize/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
tag_vectorize/strided_slicel
tag_vectorize/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/sub/x�
tag_vectorize/subSubtag_vectorize/sub/x:output:0$tag_vectorize/strided_slice:output:0*
T0*
_output_shapes
: 2
tag_vectorize/subn
tag_vectorize/Less/yConst*
_output_shapes
: *
dtype0*
value	B :2
tag_vectorize/Less/y�
tag_vectorize/LessLess$tag_vectorize/strided_slice:output:0tag_vectorize/Less/y:output:0*
T0*
_output_shapes
: 2
tag_vectorize/Less�
tag_vectorize/condStatelessIftag_vectorize/Less:z:0tag_vectorize/sub:z:0:tag_vectorize/RaggedToTensor/RaggedTensorToTensor:result:0*
Tcond0
*
Tin
2	*
Tout
2	*
_lower_using_switch_merge(*0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
else_branch!R
tag_vectorize_cond_false_2418*/
output_shapes
:������������������*/
then_branch R
tag_vectorize_cond_true_24172
tag_vectorize/cond�
tag_vectorize/cond/IdentityIdentitytag_vectorize/cond:output:0*
T0	*'
_output_shapes
:���������2
tag_vectorize/cond/Identity�
+tag_embedding_layer/StatefulPartitionedCallStatefulPartitionedCall$tag_vectorize/cond/Identity:output:0tag_embedding_layer_2456*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_tag_embedding_layer_layer_call_and_return_conditional_losses_24472-
+tag_embedding_layer/StatefulPartitionedCall�
*id_embedding_layer/StatefulPartitionedCallStatefulPartitionedCall#id_vectorize/cond/Identity:output:0id_embedding_layer_2477*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_id_embedding_layer_layer_call_and_return_conditional_losses_24682,
*id_embedding_layer/StatefulPartitionedCallp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis�
tf.concat/concatConcatV24tag_embedding_layer/StatefulPartitionedCall:output:03id_embedding_layer/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������	2
tf.concat/concat�
(global_average_pooling1d/PartitionedCallPartitionedCalltf.concat/concat:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_24872*
(global_average_pooling1d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0
dense_2516
dense_2518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_25052
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2543dense_1_2545*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_25322!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall+^id_embedding_layer/StatefulPartitionedCallF^id_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2,^tag_embedding_layer/StatefulPartitionedCallE^tag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*id_embedding_layer/StatefulPartitionedCall*id_embedding_layer/StatefulPartitionedCall2�
Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV2Eid_vectorize/string_lookup_1/None_lookup_table_find/LookupTableFindV22Z
+tag_embedding_layer/StatefulPartitionedCall+tag_embedding_layer/StatefulPartitionedCall2�
Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2Dtag_vectorize/string_lookup/None_lookup_table_find/LookupTableFindV2:K G
'
_output_shapes
:���������

_user_specified_nameid:QM
'
_output_shapes
:���������
"
_user_specified_name
item_tag:

_output_shapes
: :

_output_shapes
: 
�
�
tag_vectorize_cond_false_3061"
tag_vectorize_cond_placeholderV
Rtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor	
tag_vectorize_cond_identity	�
&tag_vectorize/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&tag_vectorize/cond/strided_slice/stack�
(tag_vectorize/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(tag_vectorize/cond/strided_slice/stack_1�
(tag_vectorize/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(tag_vectorize/cond/strided_slice/stack_2�
 tag_vectorize/cond/strided_sliceStridedSliceRtag_vectorize_cond_strided_slice_tag_vectorize_raggedtotensor_raggedtensortotensor/tag_vectorize/cond/strided_slice/stack:output:01tag_vectorize/cond/strided_slice/stack_1:output:01tag_vectorize/cond/strided_slice/stack_2:output:0*
Index0*
T0	*0
_output_shapes
:������������������*

begin_mask*
end_mask2"
 tag_vectorize/cond/strided_slice�
tag_vectorize/cond/IdentityIdentity)tag_vectorize/cond/strided_slice:output:0*
T0	*0
_output_shapes
:������������������2
tag_vectorize/cond/Identity"C
tag_vectorize_cond_identity$tag_vectorize/cond/Identity:output:0*1
_input_shapes 
: :������������������: 

_output_shapes
: :62
0
_output_shapes
:������������������
�

�
$__inference_model_layer_call_fn_3606
	inputs_id
inputs_item_tag
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	inputs_idinputs_item_tagunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_29012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:���������:���������:: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������
#
_user_specified_name	inputs/id:XT
'
_output_shapes
:���������
)
_user_specified_nameinputs/item_tag:

_output_shapes
: :

_output_shapes
: 
�
)
__inference_<lambda>_3815
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
�
x
2__inference_tag_embedding_layer_layer_call_fn_3648

inputs	
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_tag_embedding_layer_layer_call_and_return_conditional_losses_24472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_id_embedding_layer_layer_call_and_return_conditional_losses_3657

inputs	
embedding_lookup_3651
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_3651inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/3651*+
_output_shapes
:���������*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/3651*+
_output_shapes
:���������2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
__inference__creator_3731
identity��string_lookup_index_table�
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_2*
value_dtype0	2
string_lookup_index_table�
IdentityIdentity(string_lookup_index_table:table_handle:0^string_lookup_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 26
string_lookup_index_tablestring_lookup_index_table
�
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_2280

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
-
__inference__initializer_3736
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
1
id+
serving_default_id:0���������
=
item_tag1
serving_default_item_tag:0���������;
dense_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�k
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
s_default_save_signature
t__call__
*u&call_and_return_all_conditional_losses"�g
_tf_keras_network�g{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "item_tag"}, "name": "item_tag", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "id"}, "name": "id", "inbound_nodes": []}, {"class_name": "TextVectorization", "config": {"name": "tag_vectorize", "trainable": true, "dtype": "string", "max_tokens": 50, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 8, "pad_to_max_tokens": true}, "name": "tag_vectorize", "inbound_nodes": [[["item_tag", 0, 0, {}]]]}, {"class_name": "TextVectorization", "config": {"name": "id_vectorize", "trainable": true, "dtype": "string", "max_tokens": null, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 1, "pad_to_max_tokens": true}, "name": "id_vectorize", "inbound_nodes": [[["id", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "tag_embedding_layer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 51, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "tag_embedding_layer", "inbound_nodes": [[["tag_vectorize", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "id_embedding_layer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 101, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "id_embedding_layer", "inbound_nodes": [[["id_vectorize", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat", "inbound_nodes": [[["tag_embedding_layer", 0, 0, {"axis": 1}], ["id_embedding_layer", 0, 0, {"axis": 1}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d", "inbound_nodes": [[["tf.concat", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": {"id": ["id", 0, 0], "item_tag": ["item_tag", 0, 0]}, "output_layers": [["dense_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"id": {"class_name": "TensorShape", "items": [null, 1]}, "item_tag": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "item_tag"}, "name": "item_tag", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "id"}, "name": "id", "inbound_nodes": []}, {"class_name": "TextVectorization", "config": {"name": "tag_vectorize", "trainable": true, "dtype": "string", "max_tokens": 50, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 8, "pad_to_max_tokens": true}, "name": "tag_vectorize", "inbound_nodes": [[["item_tag", 0, 0, {}]]]}, {"class_name": "TextVectorization", "config": {"name": "id_vectorize", "trainable": true, "dtype": "string", "max_tokens": null, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 1, "pad_to_max_tokens": true}, "name": "id_vectorize", "inbound_nodes": [[["id", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "tag_embedding_layer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 51, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "tag_embedding_layer", "inbound_nodes": [[["tag_vectorize", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "id_embedding_layer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 101, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "id_embedding_layer", "inbound_nodes": [[["id_vectorize", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat", "inbound_nodes": [[["tag_embedding_layer", 0, 0, {"axis": 1}], ["id_embedding_layer", 0, 0, {"axis": 1}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d", "inbound_nodes": [[["tf.concat", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": {"id": ["id", 0, 0], "item_tag": ["item_tag", 0, 0]}, "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "item_tag", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "item_tag"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "id", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "id"}}
�
state_variables
_index_lookup_layer
	keras_api"�
_tf_keras_layer�{"class_name": "TextVectorization", "name": "tag_vectorize", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tag_vectorize", "trainable": true, "dtype": "string", "max_tokens": 50, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 8, "pad_to_max_tokens": true}, "build_input_shape": {"class_name": "TensorShape", "items": [39, 1]}}
�
state_variables
_index_lookup_layer
	keras_api"�
_tf_keras_layer�{"class_name": "TextVectorization", "name": "id_vectorize", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "id_vectorize", "trainable": true, "dtype": "string", "max_tokens": null, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 1, "pad_to_max_tokens": true}, "build_input_shape": {"class_name": "TensorShape", "items": [39, 1]}}
�

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "tag_embedding_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "tag_embedding_layer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 51, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
�

embeddings
	variables
trainable_variables
regularization_losses
 	keras_api
|__call__
*}&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "id_embedding_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "id_embedding_layer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 101, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
�
!	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}}
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
~__call__
*&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
�

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
�
2iter

3beta_1

4beta_2
	5decay
6learning_ratemgmh&mi'mj,mk-mlvmvn&vo'vp,vq-vr"
	optimizer
J
2
3
&4
'5
,6
-7"
trackable_list_wrapper
J
0
1
&2
'3
,4
-5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
7metrics
8layer_metrics
	variables
9non_trainable_variables

:layers
trainable_variables
regularization_losses
;layer_regularization_losses
t__call__
s_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
�
<state_variables

=_table
>	keras_api"�
_tf_keras_layer�{"class_name": "StringLookup", "name": "string_lookup", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "string", "invert": false, "max_tokens": 50, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
"
_generic_user_object
 "
trackable_dict_wrapper
�
?state_variables

@_table
A	keras_api"�
_tf_keras_layer�{"class_name": "StringLookup", "name": "string_lookup_1", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
"
_generic_user_object
0:.32tag_embedding_layer/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bmetrics
Clayer_metrics
	variables
Dnon_trainable_variables

Elayers
trainable_variables
regularization_losses
Flayer_regularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
/:-e2id_embedding_layer/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gmetrics
Hlayer_metrics
	variables
Inon_trainable_variables

Jlayers
trainable_variables
regularization_losses
Klayer_regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Lmetrics
Mlayer_metrics
"	variables
Nnon_trainable_variables

Olayers
#trainable_variables
$regularization_losses
Player_regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Qmetrics
Rlayer_metrics
(	variables
Snon_trainable_variables

Tlayers
)trainable_variables
*regularization_losses
Ulayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Vmetrics
Wlayer_metrics
.	variables
Xnon_trainable_variables

Ylayers
/trainable_variables
0regularization_losses
Zlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
[0
\1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R
�_create_resource
�_initialize
�_destroy_resourceR Z
tablevw
"
_generic_user_object
 "
trackable_dict_wrapper
R
�_create_resource
�_initialize
�_destroy_resourceR Z
tablexy
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	]total
	^count
_	variables
`	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�"
atrue_positives
btrue_negatives
cfalse_positives
dfalse_negatives
e	variables
f	keras_api"�!
_tf_keras_metric�!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
:  (2total
:  (2count
.
]0
^1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
<
a0
b1
c2
d3"
trackable_list_wrapper
-
e	variables"
_generic_user_object
5:332%Adam/tag_embedding_layer/embeddings/m
4:2e2$Adam/id_embedding_layer/embeddings/m
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
5:332%Adam/tag_embedding_layer/embeddings/v
4:2e2$Adam/id_embedding_layer/embeddings/v
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
�2�
__inference__wrapped_model_2264�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *\�Y
W�T
"
id�
id���������
.
item_tag"�
item_tag���������
�2�
$__inference_model_layer_call_fn_3124
$__inference_model_layer_call_fn_2924
$__inference_model_layer_call_fn_3606
$__inference_model_layer_call_fn_3632�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_model_layer_call_and_return_conditional_losses_2723
?__inference_model_layer_call_and_return_conditional_losses_3397
?__inference_model_layer_call_and_return_conditional_losses_2549
?__inference_model_layer_call_and_return_conditional_losses_3580�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
__inference_save_fn_3775checkpoint_key"�
���
FullArgSpec
args�
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�	
� 
�B�
__inference_restore_fn_3783restored_tensors_0restored_tensors_1"�
���
FullArgSpec
args� 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
	�	
�B�
__inference_save_fn_3802checkpoint_key"�
���
FullArgSpec
args�
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�	
� 
�B�
__inference_restore_fn_3810restored_tensors_0restored_tensors_1"�
���
FullArgSpec
args� 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
	�	
�2�
2__inference_tag_embedding_layer_layer_call_fn_3648�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_tag_embedding_layer_layer_call_and_return_conditional_losses_3641�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_id_embedding_layer_layer_call_fn_3664�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_id_embedding_layer_layer_call_and_return_conditional_losses_3657�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
7__inference_global_average_pooling1d_layer_call_fn_3675
7__inference_global_average_pooling1d_layer_call_fn_3686�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3670
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3681�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_dense_layer_call_fn_3706�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_dense_layer_call_and_return_conditional_losses_3697�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_1_layer_call_fn_3726�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_1_layer_call_and_return_conditional_losses_3717�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_3160iditem_tag"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference__creator_3731�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__initializer_3736�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__destroyer_3741�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__creator_3746�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__initializer_3751�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__destroyer_3756�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
	J
Const
J	
Const_15
__inference__creator_3731�

� 
� "� 5
__inference__creator_3746�

� 
� "� 7
__inference__destroyer_3741�

� 
� "� 7
__inference__destroyer_3756�

� 
� "� 9
__inference__initializer_3736�

� 
� "� 9
__inference__initializer_3751�

� 
� "� �
__inference__wrapped_model_2264�@�=�&',-f�c
\�Y
W�T
"
id�
id���������
.
item_tag"�
item_tag���������
� "1�.
,
dense_1!�
dense_1����������
A__inference_dense_1_layer_call_and_return_conditional_losses_3717\,-/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� y
&__inference_dense_1_layer_call_fn_3726O,-/�,
%�"
 �
inputs���������
� "�����������
?__inference_dense_layer_call_and_return_conditional_losses_3697\&'/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� w
$__inference_dense_layer_call_fn_3706O&'/�,
%�"
 �
inputs���������
� "�����������
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3670`7�4
-�*
$�!
inputs���������	

 
� "%�"
�
0���������
� �
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3681{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
7__inference_global_average_pooling1d_layer_call_fn_3675S7�4
-�*
$�!
inputs���������	

 
� "�����������
7__inference_global_average_pooling1d_layer_call_fn_3686nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
L__inference_id_embedding_layer_layer_call_and_return_conditional_losses_3657_/�,
%�"
 �
inputs���������	
� ")�&
�
0���������
� �
1__inference_id_embedding_layer_layer_call_fn_3664R/�,
%�"
 �
inputs���������	
� "�����������
?__inference_model_layer_call_and_return_conditional_losses_2549�@�=�&',-n�k
d�a
W�T
"
id�
id���������
.
item_tag"�
item_tag���������
p

 
� "%�"
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_2723�@�=�&',-n�k
d�a
W�T
"
id�
id���������
.
item_tag"�
item_tag���������
p 

 
� "%�"
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_3397�@�=�&',-|�y
r�o
e�b
)
id#� 
	inputs/id���������
5
item_tag)�&
inputs/item_tag���������
p

 
� "%�"
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_3580�@�=�&',-|�y
r�o
e�b
)
id#� 
	inputs/id���������
5
item_tag)�&
inputs/item_tag���������
p 

 
� "%�"
�
0���������
� �
$__inference_model_layer_call_fn_2924�@�=�&',-n�k
d�a
W�T
"
id�
id���������
.
item_tag"�
item_tag���������
p

 
� "�����������
$__inference_model_layer_call_fn_3124�@�=�&',-n�k
d�a
W�T
"
id�
id���������
.
item_tag"�
item_tag���������
p 

 
� "�����������
$__inference_model_layer_call_fn_3606�@�=�&',-|�y
r�o
e�b
)
id#� 
	inputs/id���������
5
item_tag)�&
inputs/item_tag���������
p

 
� "�����������
$__inference_model_layer_call_fn_3632�@�=�&',-|�y
r�o
e�b
)
id#� 
	inputs/id���������
5
item_tag)�&
inputs/item_tag���������
p 

 
� "����������x
__inference_restore_fn_3783Y=K�H
A�>
�
restored_tensors_0
�
restored_tensors_1	
� "� x
__inference_restore_fn_3810Y@K�H
A�>
�
restored_tensors_0
�
restored_tensors_1	
� "� �
__inference_save_fn_3775�=&�#
�
�
checkpoint_key 
� "���
`�]

name�
0/name 
#

slice_spec�
0/slice_spec 

tensor�
0/tensor
`�]

name�
1/name 
#

slice_spec�
1/slice_spec 

tensor�
1/tensor	�
__inference_save_fn_3802�@&�#
�
�
checkpoint_key 
� "���
`�]

name�
0/name 
#

slice_spec�
0/slice_spec 

tensor�
0/tensor
`�]

name�
1/name 
#

slice_spec�
1/slice_spec 

tensor�
1/tensor	�
"__inference_signature_wrapper_3160�@�=�&',-a�^
� 
W�T
"
id�
id���������
.
item_tag"�
item_tag���������"1�.
,
dense_1!�
dense_1����������
M__inference_tag_embedding_layer_layer_call_and_return_conditional_losses_3641_/�,
%�"
 �
inputs���������	
� ")�&
�
0���������
� �
2__inference_tag_embedding_layer_layer_call_fn_3648R/�,
%�"
 �
inputs���������	
� "����������