ыф
гЈ
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
3
Square
x"T
y"T"
Ttype:
2
	
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8сё

user_id_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
рд*-
shared_nameuser_id_embedding/embeddings

0user_id_embedding/embeddings/Read/ReadVariableOpReadVariableOpuser_id_embedding/embeddings* 
_output_shapes
:
рд*
dtype0

 member_type_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" member_type_embedding/embeddings

4member_type_embedding/embeddings/Read/ReadVariableOpReadVariableOp member_type_embedding/embeddings*
_output_shapes

:*
dtype0

user_type_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name user_type_embedding/embeddings

2user_type_embedding/embeddings/Read/ReadVariableOpReadVariableOpuser_type_embedding/embeddings*
_output_shapes

:*
dtype0

item_id_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*-
shared_nameitem_id_embedding/embeddings

0item_id_embedding/embeddings/Read/ReadVariableOpReadVariableOpitem_id_embedding/embeddings*
_output_shapes
:	Ќ*
dtype0

!item_catalog_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!item_catalog_embedding/embeddings

5item_catalog_embedding/embeddings/Read/ReadVariableOpReadVariableOp!item_catalog_embedding/embeddings*
_output_shapes

:*
dtype0

item_tag_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	­*.
shared_nameitem_tag_embedding/embeddings

1item_tag_embedding/embeddings/Read/ReadVariableOpReadVariableOpitem_tag_embedding/embeddings*
_output_shapes
:	­*
dtype0

dnn_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namednn_layer1/kernel
y
%dnn_layer1/kernel/Read/ReadVariableOpReadVariableOpdnn_layer1/kernel* 
_output_shapes
:
*
dtype0
w
dnn_layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namednn_layer1/bias
p
#dnn_layer1/bias/Read/ReadVariableOpReadVariableOpdnn_layer1/bias*
_output_shapes	
:*
dtype0

dnn_layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namednn_layer2/kernel
y
%dnn_layer2/kernel/Read/ReadVariableOpReadVariableOpdnn_layer2/kernel* 
_output_shapes
:
*
dtype0
w
dnn_layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namednn_layer2/bias
p
#dnn_layer2/bias/Read/ReadVariableOpReadVariableOpdnn_layer2/bias*
_output_shapes	
:*
dtype0

dnn_layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namednn_layer3/kernel
y
%dnn_layer3/kernel/Read/ReadVariableOpReadVariableOpdnn_layer3/kernel* 
_output_shapes
:
*
dtype0
w
dnn_layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namednn_layer3/bias
p
#dnn_layer3/bias/Read/ReadVariableOpReadVariableOpdnn_layer3/bias*
_output_shapes	
:*
dtype0
}
fm_linear/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namefm_linear/kernel
v
$fm_linear/kernel/Read/ReadVariableOpReadVariableOpfm_linear/kernel*
_output_shapes
:	*
dtype0
t
fm_linear/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefm_linear/bias
m
"fm_linear/bias/Read/ReadVariableOpReadVariableOpfm_linear/bias*
_output_shapes
:*
dtype0

dnn_layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namednn_layer4/kernel
x
%dnn_layer4/kernel/Read/ReadVariableOpReadVariableOpdnn_layer4/kernel*
_output_shapes
:	*
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
shape:Ш*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:Ш*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:Ш*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:Ш*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:Ш*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
z
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:*
dtype0
Є
#Adam/user_id_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
рд*4
shared_name%#Adam/user_id_embedding/embeddings/m

7Adam/user_id_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp#Adam/user_id_embedding/embeddings/m* 
_output_shapes
:
рд*
dtype0
Њ
'Adam/member_type_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/member_type_embedding/embeddings/m
Ѓ
;Adam/member_type_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp'Adam/member_type_embedding/embeddings/m*
_output_shapes

:*
dtype0
І
%Adam/user_type_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%Adam/user_type_embedding/embeddings/m

9Adam/user_type_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp%Adam/user_type_embedding/embeddings/m*
_output_shapes

:*
dtype0
Ѓ
#Adam/item_id_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*4
shared_name%#Adam/item_id_embedding/embeddings/m

7Adam/item_id_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp#Adam/item_id_embedding/embeddings/m*
_output_shapes
:	Ќ*
dtype0
Ќ
(Adam/item_catalog_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/item_catalog_embedding/embeddings/m
Ѕ
<Adam/item_catalog_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp(Adam/item_catalog_embedding/embeddings/m*
_output_shapes

:*
dtype0
Ѕ
$Adam/item_tag_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	­*5
shared_name&$Adam/item_tag_embedding/embeddings/m

8Adam/item_tag_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp$Adam/item_tag_embedding/embeddings/m*
_output_shapes
:	­*
dtype0

Adam/dnn_layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dnn_layer1/kernel/m

,Adam/dnn_layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dnn_layer1/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dnn_layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dnn_layer1/bias/m
~
*Adam/dnn_layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dnn_layer1/bias/m*
_output_shapes	
:*
dtype0

Adam/dnn_layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dnn_layer2/kernel/m

,Adam/dnn_layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dnn_layer2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dnn_layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dnn_layer2/bias/m
~
*Adam/dnn_layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dnn_layer2/bias/m*
_output_shapes	
:*
dtype0

Adam/dnn_layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dnn_layer3/kernel/m

,Adam/dnn_layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dnn_layer3/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dnn_layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dnn_layer3/bias/m
~
*Adam/dnn_layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dnn_layer3/bias/m*
_output_shapes	
:*
dtype0

Adam/fm_linear/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/fm_linear/kernel/m

+Adam/fm_linear/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fm_linear/kernel/m*
_output_shapes
:	*
dtype0

Adam/fm_linear/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/fm_linear/bias/m
{
)Adam/fm_linear/bias/m/Read/ReadVariableOpReadVariableOpAdam/fm_linear/bias/m*
_output_shapes
:*
dtype0

Adam/dnn_layer4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/dnn_layer4/kernel/m

,Adam/dnn_layer4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dnn_layer4/kernel/m*
_output_shapes
:	*
dtype0
Є
#Adam/user_id_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
рд*4
shared_name%#Adam/user_id_embedding/embeddings/v

7Adam/user_id_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp#Adam/user_id_embedding/embeddings/v* 
_output_shapes
:
рд*
dtype0
Њ
'Adam/member_type_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/member_type_embedding/embeddings/v
Ѓ
;Adam/member_type_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp'Adam/member_type_embedding/embeddings/v*
_output_shapes

:*
dtype0
І
%Adam/user_type_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%Adam/user_type_embedding/embeddings/v

9Adam/user_type_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp%Adam/user_type_embedding/embeddings/v*
_output_shapes

:*
dtype0
Ѓ
#Adam/item_id_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ќ*4
shared_name%#Adam/item_id_embedding/embeddings/v

7Adam/item_id_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp#Adam/item_id_embedding/embeddings/v*
_output_shapes
:	Ќ*
dtype0
Ќ
(Adam/item_catalog_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(Adam/item_catalog_embedding/embeddings/v
Ѕ
<Adam/item_catalog_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp(Adam/item_catalog_embedding/embeddings/v*
_output_shapes

:*
dtype0
Ѕ
$Adam/item_tag_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	­*5
shared_name&$Adam/item_tag_embedding/embeddings/v

8Adam/item_tag_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp$Adam/item_tag_embedding/embeddings/v*
_output_shapes
:	­*
dtype0

Adam/dnn_layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dnn_layer1/kernel/v

,Adam/dnn_layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dnn_layer1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dnn_layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dnn_layer1/bias/v
~
*Adam/dnn_layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dnn_layer1/bias/v*
_output_shapes	
:*
dtype0

Adam/dnn_layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dnn_layer2/kernel/v

,Adam/dnn_layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dnn_layer2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dnn_layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dnn_layer2/bias/v
~
*Adam/dnn_layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dnn_layer2/bias/v*
_output_shapes	
:*
dtype0

Adam/dnn_layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/dnn_layer3/kernel/v

,Adam/dnn_layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dnn_layer3/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dnn_layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dnn_layer3/bias/v
~
*Adam/dnn_layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dnn_layer3/bias/v*
_output_shapes	
:*
dtype0

Adam/fm_linear/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/fm_linear/kernel/v

+Adam/fm_linear/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fm_linear/kernel/v*
_output_shapes
:	*
dtype0

Adam/fm_linear/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/fm_linear/bias/v
{
)Adam/fm_linear/bias/v/Read/ReadVariableOpReadVariableOpAdam/fm_linear/bias/v*
_output_shapes
:*
dtype0

Adam/dnn_layer4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/dnn_layer4/kernel/v

,Adam/dnn_layer4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dnn_layer4/kernel/v*
_output_shapes
:	*
dtype0

NoOpNoOp
x
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*вw
valueШwBХw BОw

layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer_with_weights-9
layer-20
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer-25
layer-26
	optimizer
regularization_losses
	variables
trainable_variables
 	keras_api
!
signatures
 
 
 
 
 
 
b
"
embeddings
#regularization_losses
$	variables
%trainable_variables
&	keras_api
b
'
embeddings
(regularization_losses
)	variables
*trainable_variables
+	keras_api
b
,
embeddings
-regularization_losses
.	variables
/trainable_variables
0	keras_api
b
1
embeddings
2regularization_losses
3	variables
4trainable_variables
5	keras_api
b
6
embeddings
7regularization_losses
8	variables
9trainable_variables
:	keras_api
b
;
embeddings
<regularization_losses
=	variables
>trainable_variables
?	keras_api
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
R
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
h

Hkernel
Ibias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
h

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
R
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
h

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
h

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
R
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
R
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
R
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
^

xkernel
yregularization_losses
z	variables
{trainable_variables
|	keras_api
S
}regularization_losses
~	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
ё
	iter
beta_1
beta_2

decay
learning_rate"m'm,m1m6m;mHmImRmSm`mamfmgmxm"v'v,v1v6v;vHvIv RvЁSvЂ`vЃavЄfvЅgvІxvЇ
 
n
"0
'1
,2
13
64
;5
H6
I7
R8
S9
`10
a11
f12
g13
x14
n
"0
'1
,2
13
64
;5
H6
I7
R8
S9
`10
a11
f12
g13
x14
В
metrics
layers
regularization_losses
	variables
layer_metrics
trainable_variables
 layer_regularization_losses
non_trainable_variables
 
lj
VARIABLE_VALUEuser_id_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

"0

"0
В
metrics
#regularization_losses
$	variables
 layer_regularization_losses
layer_metrics
%trainable_variables
layers
non_trainable_variables
pn
VARIABLE_VALUE member_type_embedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

'0

'0
В
metrics
(regularization_losses
)	variables
 layer_regularization_losses
layer_metrics
*trainable_variables
layers
non_trainable_variables
nl
VARIABLE_VALUEuser_type_embedding/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

,0

,0
В
metrics
-regularization_losses
.	variables
 layer_regularization_losses
layer_metrics
/trainable_variables
layers
non_trainable_variables
lj
VARIABLE_VALUEitem_id_embedding/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

10

10
В
metrics
2regularization_losses
3	variables
 layer_regularization_losses
 layer_metrics
4trainable_variables
Ёlayers
Ђnon_trainable_variables
qo
VARIABLE_VALUE!item_catalog_embedding/embeddings:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

60

60
В
Ѓmetrics
7regularization_losses
8	variables
 Єlayer_regularization_losses
Ѕlayer_metrics
9trainable_variables
Іlayers
Їnon_trainable_variables
mk
VARIABLE_VALUEitem_tag_embedding/embeddings:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

;0

;0
В
Јmetrics
<regularization_losses
=	variables
 Љlayer_regularization_losses
Њlayer_metrics
>trainable_variables
Ћlayers
Ќnon_trainable_variables
 
 
 
В
­metrics
@regularization_losses
A	variables
 Ўlayer_regularization_losses
Џlayer_metrics
Btrainable_variables
Аlayers
Бnon_trainable_variables
 
 
 
В
Вmetrics
Dregularization_losses
E	variables
 Гlayer_regularization_losses
Дlayer_metrics
Ftrainable_variables
Еlayers
Жnon_trainable_variables
][
VARIABLE_VALUEdnn_layer1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdnn_layer1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1

H0
I1
В
Зmetrics
Jregularization_losses
K	variables
 Иlayer_regularization_losses
Йlayer_metrics
Ltrainable_variables
Кlayers
Лnon_trainable_variables
 
 
 
В
Мmetrics
Nregularization_losses
O	variables
 Нlayer_regularization_losses
Оlayer_metrics
Ptrainable_variables
Пlayers
Рnon_trainable_variables
][
VARIABLE_VALUEdnn_layer2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdnn_layer2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
В
Сmetrics
Tregularization_losses
U	variables
 Тlayer_regularization_losses
Уlayer_metrics
Vtrainable_variables
Фlayers
Хnon_trainable_variables
 
 
 
В
Цmetrics
Xregularization_losses
Y	variables
 Чlayer_regularization_losses
Шlayer_metrics
Ztrainable_variables
Щlayers
Ъnon_trainable_variables
 
 
 
В
Ыmetrics
\regularization_losses
]	variables
 Ьlayer_regularization_losses
Эlayer_metrics
^trainable_variables
Юlayers
Яnon_trainable_variables
][
VARIABLE_VALUEdnn_layer3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdnn_layer3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1

`0
a1
В
аmetrics
bregularization_losses
c	variables
 бlayer_regularization_losses
вlayer_metrics
dtrainable_variables
гlayers
дnon_trainable_variables
\Z
VARIABLE_VALUEfm_linear/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEfm_linear/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1

f0
g1
В
еmetrics
hregularization_losses
i	variables
 жlayer_regularization_losses
зlayer_metrics
jtrainable_variables
иlayers
йnon_trainable_variables
 
 
 
В
кmetrics
lregularization_losses
m	variables
 лlayer_regularization_losses
мlayer_metrics
ntrainable_variables
нlayers
оnon_trainable_variables
 
 
 
В
пmetrics
pregularization_losses
q	variables
 рlayer_regularization_losses
сlayer_metrics
rtrainable_variables
тlayers
уnon_trainable_variables
 
 
 
В
фmetrics
tregularization_losses
u	variables
 хlayer_regularization_losses
цlayer_metrics
vtrainable_variables
чlayers
шnon_trainable_variables
^\
VARIABLE_VALUEdnn_layer4/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

x0

x0
В
щmetrics
yregularization_losses
z	variables
 ъlayer_regularization_losses
ыlayer_metrics
{trainable_variables
ьlayers
эnon_trainable_variables
 
 
 
В
юmetrics
}regularization_losses
~	variables
 яlayer_regularization_losses
№layer_metrics
trainable_variables
ёlayers
ђnon_trainable_variables
 
 
 
Е
ѓmetrics
regularization_losses
	variables
 єlayer_regularization_losses
ѕlayer_metrics
trainable_variables
іlayers
їnon_trainable_variables
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

ј0
љ1
њ2
Ю
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
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
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
 
 
 
 
8

ћtotal

ќcount
§	variables
ў	keras_api
v
џtrue_positives
true_negatives
false_positives
false_negatives
	variables
	keras_api
\

thresholds
true_positives
false_negatives
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ћ0
ќ1

§	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
џ0
1
2
3

	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables

VARIABLE_VALUE#Adam/user_id_embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'Adam/member_type_embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%Adam/user_type_embedding/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/item_id_embedding/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/item_catalog_embedding/embeddings/mVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/item_tag_embedding/embeddings/mVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dnn_layer1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dnn_layer1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dnn_layer2/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dnn_layer2/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dnn_layer3/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dnn_layer3/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/fm_linear/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/fm_linear/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dnn_layer4/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/user_id_embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'Adam/member_type_embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%Adam/user_type_embedding/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/item_id_embedding/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Adam/item_catalog_embedding/embeddings/vVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/item_tag_embedding/embeddings/vVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dnn_layer1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dnn_layer1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dnn_layer2/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dnn_layer2/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dnn_layer3/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dnn_layer3/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/fm_linear/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/fm_linear/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dnn_layer4/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_item_catalogPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
z
serving_default_item_idPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
{
serving_default_item_tagPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
~
serving_default_member_typePlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
z
serving_default_user_idPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
|
serving_default_user_typePlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
С
StatefulPartitionedCallStatefulPartitionedCallserving_default_item_catalogserving_default_item_idserving_default_item_tagserving_default_member_typeserving_default_user_idserving_default_user_typeuser_id_embedding/embeddings member_type_embedding/embeddingsuser_type_embedding/embeddingsitem_id_embedding/embeddings!item_catalog_embedding/embeddingsitem_tag_embedding/embeddingsdnn_layer1/kerneldnn_layer1/biasdnn_layer2/kerneldnn_layer2/biasdnn_layer3/kerneldnn_layer3/biasfm_linear/kernelfm_linear/biasdnn_layer4/kernel* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_75650
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0user_id_embedding/embeddings/Read/ReadVariableOp4member_type_embedding/embeddings/Read/ReadVariableOp2user_type_embedding/embeddings/Read/ReadVariableOp0item_id_embedding/embeddings/Read/ReadVariableOp5item_catalog_embedding/embeddings/Read/ReadVariableOp1item_tag_embedding/embeddings/Read/ReadVariableOp%dnn_layer1/kernel/Read/ReadVariableOp#dnn_layer1/bias/Read/ReadVariableOp%dnn_layer2/kernel/Read/ReadVariableOp#dnn_layer2/bias/Read/ReadVariableOp%dnn_layer3/kernel/Read/ReadVariableOp#dnn_layer3/bias/Read/ReadVariableOp$fm_linear/kernel/Read/ReadVariableOp"fm_linear/bias/Read/ReadVariableOp%dnn_layer4/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp7Adam/user_id_embedding/embeddings/m/Read/ReadVariableOp;Adam/member_type_embedding/embeddings/m/Read/ReadVariableOp9Adam/user_type_embedding/embeddings/m/Read/ReadVariableOp7Adam/item_id_embedding/embeddings/m/Read/ReadVariableOp<Adam/item_catalog_embedding/embeddings/m/Read/ReadVariableOp8Adam/item_tag_embedding/embeddings/m/Read/ReadVariableOp,Adam/dnn_layer1/kernel/m/Read/ReadVariableOp*Adam/dnn_layer1/bias/m/Read/ReadVariableOp,Adam/dnn_layer2/kernel/m/Read/ReadVariableOp*Adam/dnn_layer2/bias/m/Read/ReadVariableOp,Adam/dnn_layer3/kernel/m/Read/ReadVariableOp*Adam/dnn_layer3/bias/m/Read/ReadVariableOp+Adam/fm_linear/kernel/m/Read/ReadVariableOp)Adam/fm_linear/bias/m/Read/ReadVariableOp,Adam/dnn_layer4/kernel/m/Read/ReadVariableOp7Adam/user_id_embedding/embeddings/v/Read/ReadVariableOp;Adam/member_type_embedding/embeddings/v/Read/ReadVariableOp9Adam/user_type_embedding/embeddings/v/Read/ReadVariableOp7Adam/item_id_embedding/embeddings/v/Read/ReadVariableOp<Adam/item_catalog_embedding/embeddings/v/Read/ReadVariableOp8Adam/item_tag_embedding/embeddings/v/Read/ReadVariableOp,Adam/dnn_layer1/kernel/v/Read/ReadVariableOp*Adam/dnn_layer1/bias/v/Read/ReadVariableOp,Adam/dnn_layer2/kernel/v/Read/ReadVariableOp*Adam/dnn_layer2/bias/v/Read/ReadVariableOp,Adam/dnn_layer3/kernel/v/Read/ReadVariableOp*Adam/dnn_layer3/bias/v/Read/ReadVariableOp+Adam/fm_linear/kernel/v/Read/ReadVariableOp)Adam/fm_linear/bias/v/Read/ReadVariableOp,Adam/dnn_layer4/kernel/v/Read/ReadVariableOpConst*G
Tin@
>2<	*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_76819

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameuser_id_embedding/embeddings member_type_embedding/embeddingsuser_type_embedding/embeddingsitem_id_embedding/embeddings!item_catalog_embedding/embeddingsitem_tag_embedding/embeddingsdnn_layer1/kerneldnn_layer1/biasdnn_layer2/kerneldnn_layer2/biasdnn_layer3/kerneldnn_layer3/biasfm_linear/kernelfm_linear/biasdnn_layer4/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1false_negatives_1#Adam/user_id_embedding/embeddings/m'Adam/member_type_embedding/embeddings/m%Adam/user_type_embedding/embeddings/m#Adam/item_id_embedding/embeddings/m(Adam/item_catalog_embedding/embeddings/m$Adam/item_tag_embedding/embeddings/mAdam/dnn_layer1/kernel/mAdam/dnn_layer1/bias/mAdam/dnn_layer2/kernel/mAdam/dnn_layer2/bias/mAdam/dnn_layer3/kernel/mAdam/dnn_layer3/bias/mAdam/fm_linear/kernel/mAdam/fm_linear/bias/mAdam/dnn_layer4/kernel/m#Adam/user_id_embedding/embeddings/v'Adam/member_type_embedding/embeddings/v%Adam/user_type_embedding/embeddings/v#Adam/item_id_embedding/embeddings/v(Adam/item_catalog_embedding/embeddings/v$Adam/item_tag_embedding/embeddings/vAdam/dnn_layer1/kernel/vAdam/dnn_layer1/bias/vAdam/dnn_layer2/kernel/vAdam/dnn_layer2/bias/vAdam/dnn_layer3/kernel/vAdam/dnn_layer3/bias/vAdam/fm_linear/kernel/vAdam/fm_linear/bias/vAdam/dnn_layer4/kernel/v*F
Tin?
=2;*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_77003§Ы

Ш
E__inference_dnn_layer1_layer_call_and_return_conditional_losses_74741

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ1dnn_layer1/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluЫ
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer1/kernel/Regularizer/SquareSquare;dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer1/kernel/Regularizer/Square
#dnn_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer1/kernel/Regularizer/ConstЦ
!dnn_layer1/kernel/Regularizer/SumSum(dnn_layer1/kernel/Regularizer/Square:y:0,dnn_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/Sum
#dnn_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer1/kernel/Regularizer/mul/xШ
!dnn_layer1/kernel/Regularizer/mulMul,dnn_layer1/kernel/Regularizer/mul/x:output:0*dnn_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/mulУ
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer1/bias/Regularizer/SquareSquare9dnn_layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer1/bias/Regularizer/Square
!dnn_layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer1/bias/Regularizer/ConstО
dnn_layer1/bias/Regularizer/SumSum&dnn_layer1/bias/Regularizer/Square:y:0*dnn_layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/Sum
!dnn_layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer1/bias/Regularizer/mul/xР
dnn_layer1/bias/Regularizer/mulMul*dnn_layer1/bias/Regularizer/mul/x:output:0(dnn_layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/mul
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dnn_layer1/bias/Regularizer/Square/ReadVariableOp4^dnn_layer1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dnn_layer1/bias/Regularizer/Square/ReadVariableOp1dnn_layer1/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

j
@__inference_add_8_layer_call_and_return_conditional_losses_75046

inputs
inputs_1
identityW
addAddV2inputsinputs_1*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л
y
3__inference_user_type_embedding_layer_call_fn_76091

inputs
unknown
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_user_type_embedding_layer_call_and_return_conditional_losses_746052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с
|
6__inference_item_catalog_embedding_layer_call_fn_76123

inputs
unknown
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_item_catalog_embedding_layer_call_and_return_conditional_losses_746472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

C
'__inference_sigmoid_layer_call_fn_76518

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_sigmoid_layer_call_and_return_conditional_losses_750602
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
F
*__inference_flatten_17_layer_call_fn_76171

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_17_layer_call_and_return_conditional_losses_747102
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ш
E__inference_dnn_layer2_layer_call_and_return_conditional_losses_74810

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ1dnn_layer2/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluЫ
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer2/kernel/Regularizer/SquareSquare;dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer2/kernel/Regularizer/Square
#dnn_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer2/kernel/Regularizer/ConstЦ
!dnn_layer2/kernel/Regularizer/SumSum(dnn_layer2/kernel/Regularizer/Square:y:0,dnn_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/Sum
#dnn_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer2/kernel/Regularizer/mul/xШ
!dnn_layer2/kernel/Regularizer/mulMul,dnn_layer2/kernel/Regularizer/mul/x:output:0*dnn_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/mulУ
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer2/bias/Regularizer/SquareSquare9dnn_layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer2/bias/Regularizer/Square
!dnn_layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer2/bias/Regularizer/ConstО
dnn_layer2/bias/Regularizer/SumSum&dnn_layer2/bias/Regularizer/Square:y:0*dnn_layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/Sum
!dnn_layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer2/bias/Regularizer/mul/xР
dnn_layer2/bias/Regularizer/mulMul*dnn_layer2/bias/Regularizer/mul/x:output:0(dnn_layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/mul
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dnn_layer2/bias/Regularizer/Square/ReadVariableOp4^dnn_layer2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dnn_layer2/bias/Regularizer/Square/ReadVariableOp1dnn_layer2/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з
w
1__inference_item_id_embedding_layer_call_fn_76107

inputs
unknown
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_item_id_embedding_layer_call_and_return_conditional_losses_746262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
E__inference_dropout_24_layer_call_and_return_conditional_losses_74769

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь
c
E__inference_dropout_24_layer_call_and_return_conditional_losses_76232

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ђ
__inference_loss_fn_3_76562>
:dnn_layer2_bias_regularizer_square_readvariableop_resource
identityЂ1dnn_layer2/bias/Regularizer/Square/ReadVariableOpо
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOp:dnn_layer2_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer2/bias/Regularizer/SquareSquare9dnn_layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer2/bias/Regularizer/Square
!dnn_layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer2/bias/Regularizer/ConstО
dnn_layer2/bias/Regularizer/SumSum&dnn_layer2/bias/Regularizer/Square:y:0*dnn_layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/Sum
!dnn_layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer2/bias/Regularizer/mul/xР
dnn_layer2/bias/Regularizer/mulMul*dnn_layer2/bias/Regularizer/mul/x:output:0(dnn_layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/mul
IdentityIdentity#dnn_layer2/bias/Regularizer/mul:z:02^dnn_layer2/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dnn_layer2/bias/Regularizer/Square/ReadVariableOp1dnn_layer2/bias/Regularizer/Square/ReadVariableOp

Ш
E__inference_dnn_layer2_layer_call_and_return_conditional_losses_76277

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ1dnn_layer2/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluЫ
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer2/kernel/Regularizer/SquareSquare;dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer2/kernel/Regularizer/Square
#dnn_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer2/kernel/Regularizer/ConstЦ
!dnn_layer2/kernel/Regularizer/SumSum(dnn_layer2/kernel/Regularizer/Square:y:0,dnn_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/Sum
#dnn_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer2/kernel/Regularizer/mul/xШ
!dnn_layer2/kernel/Regularizer/mulMul,dnn_layer2/kernel/Regularizer/mul/x:output:0*dnn_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/mulУ
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer2/bias/Regularizer/SquareSquare9dnn_layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer2/bias/Regularizer/Square
!dnn_layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer2/bias/Regularizer/ConstО
dnn_layer2/bias/Regularizer/SumSum&dnn_layer2/bias/Regularizer/Square:y:0*dnn_layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/Sum
!dnn_layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer2/bias/Regularizer/mul/xР
dnn_layer2/bias/Regularizer/mulMul*dnn_layer2/bias/Regularizer/mul/x:output:0(dnn_layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/mul
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dnn_layer2/bias/Regularizer/Square/ReadVariableOp4^dnn_layer2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dnn_layer2/bias/Regularizer/Square/ReadVariableOp1dnn_layer2/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

F
*__inference_dropout_24_layer_call_fn_76242

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_747742
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
c
*__inference_dropout_24_layer_call_fn_76237

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_747692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
a
E__inference_flatten_17_layer_call_and_return_conditional_losses_74710

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ

L__inference_user_id_embedding_layer_call_and_return_conditional_losses_74563

inputs
embedding_lookup_74557
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_74557inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/74557*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/74557*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	

P__inference_member_type_embedding_layer_call_and_return_conditional_losses_76068

inputs
embedding_lookup_76062
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_76062inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/76062*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/76062*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џЙ
Ю
C__inference_model_17_layer_call_and_return_conditional_losses_75513

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
user_id_embedding_75407
member_type_embedding_75410
user_type_embedding_75413
item_id_embedding_75416 
item_catalog_embedding_75419
item_tag_embedding_75422
dnn_layer1_75427
dnn_layer1_75429
dnn_layer2_75433
dnn_layer2_75435
dnn_layer3_75439
dnn_layer3_75441
fm_linear_75446
fm_linear_75448
dnn_layer4_75453
identityЂ"dnn_layer1/StatefulPartitionedCallЂ1dnn_layer1/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer2/StatefulPartitionedCallЂ1dnn_layer2/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer3/StatefulPartitionedCallЂ1dnn_layer3/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer4/StatefulPartitionedCallЂ3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpЂ!fm_linear/StatefulPartitionedCallЂ0fm_linear/bias/Regularizer/Square/ReadVariableOpЂ2fm_linear/kernel/Regularizer/Square/ReadVariableOpЂ.item_catalog_embedding/StatefulPartitionedCallЂ)item_id_embedding/StatefulPartitionedCallЂ*item_tag_embedding/StatefulPartitionedCallЂ-member_type_embedding/StatefulPartitionedCallЂ)user_id_embedding/StatefulPartitionedCallЂ+user_type_embedding/StatefulPartitionedCallЊ
)user_id_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_id_embedding_75407*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_user_id_embedding_layer_call_and_return_conditional_losses_745632+
)user_id_embedding/StatefulPartitionedCallМ
-member_type_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_2member_type_embedding_75410*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_member_type_embedding_layer_call_and_return_conditional_losses_745842/
-member_type_embedding/StatefulPartitionedCallД
+user_type_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1user_type_embedding_75413*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_user_type_embedding_layer_call_and_return_conditional_losses_746052-
+user_type_embedding/StatefulPartitionedCallЌ
)item_id_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_3item_id_embedding_75416*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_item_id_embedding_layer_call_and_return_conditional_losses_746262+
)item_id_embedding/StatefulPartitionedCallР
.item_catalog_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_4item_catalog_embedding_75419*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_item_catalog_embedding_layer_call_and_return_conditional_losses_7464720
.item_catalog_embedding/StatefulPartitionedCallА
*item_tag_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_5item_tag_embedding_75422*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_item_tag_embedding_layer_call_and_return_conditional_losses_746682,
*item_tag_embedding/StatefulPartitionedCallТ
%embedding_concatenate/PartitionedCallPartitionedCall2user_id_embedding/StatefulPartitionedCall:output:06member_type_embedding/StatefulPartitionedCall:output:04user_type_embedding/StatefulPartitionedCall:output:02item_id_embedding/StatefulPartitionedCall:output:07item_catalog_embedding/StatefulPartitionedCall:output:03item_tag_embedding/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_embedding_concatenate_layer_call_and_return_conditional_losses_746912'
%embedding_concatenate/PartitionedCall
flatten_17/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_17_layer_call_and_return_conditional_losses_747102
flatten_17/PartitionedCallМ
"dnn_layer1/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dnn_layer1_75427dnn_layer1_75429*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer1_layer_call_and_return_conditional_losses_747412$
"dnn_layer1/StatefulPartitionedCall
dropout_24/PartitionedCallPartitionedCall+dnn_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_747742
dropout_24/PartitionedCallМ
"dnn_layer2/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0dnn_layer2_75433dnn_layer2_75435*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer2_layer_call_and_return_conditional_losses_748102$
"dnn_layer2/StatefulPartitionedCall
dropout_25/PartitionedCallPartitionedCall+dnn_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_748432
dropout_25/PartitionedCallМ
"dnn_layer3/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dnn_layer3_75439dnn_layer3_75441*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer3_layer_call_and_return_conditional_losses_748792$
"dnn_layer3/StatefulPartitionedCall
flatten_16/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_749012
flatten_16/PartitionedCall
dropout_26/PartitionedCallPartitionedCall+dnn_layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_26_layer_call_and_return_conditional_losses_749262
dropout_26/PartitionedCallЖ
!fm_linear/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0fm_linear_75446fm_linear_75448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_fm_linear_layer_call_and_return_conditional_losses_749612#
!fm_linear/StatefulPartitionedCallў
fm_cross/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fm_cross_layer_call_and_return_conditional_losses_749922
fm_cross/PartitionedCallЄ
fm_combine/PartitionedCallPartitionedCall*fm_linear/StatefulPartitionedCall:output:0!fm_cross/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fm_combine_layer_call_and_return_conditional_losses_750062
fm_combine/PartitionedCallЇ
"dnn_layer4/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0dnn_layer4_75453*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer4_layer_call_and_return_conditional_losses_750282$
"dnn_layer4/StatefulPartitionedCall
add_8/PartitionedCallPartitionedCall#fm_combine/PartitionedCall:output:0+dnn_layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_8_layer_call_and_return_conditional_losses_750462
add_8/PartitionedCallы
sigmoid/PartitionedCallPartitionedCalladd_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_sigmoid_layer_call_and_return_conditional_losses_750602
sigmoid/PartitionedCallН
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer1_75427* 
_output_shapes
:
*
dtype025
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer1/kernel/Regularizer/SquareSquare;dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer1/kernel/Regularizer/Square
#dnn_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer1/kernel/Regularizer/ConstЦ
!dnn_layer1/kernel/Regularizer/SumSum(dnn_layer1/kernel/Regularizer/Square:y:0,dnn_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/Sum
#dnn_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer1/kernel/Regularizer/mul/xШ
!dnn_layer1/kernel/Regularizer/mulMul,dnn_layer1/kernel/Regularizer/mul/x:output:0*dnn_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/mulД
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer1_75429*
_output_shapes	
:*
dtype023
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer1/bias/Regularizer/SquareSquare9dnn_layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer1/bias/Regularizer/Square
!dnn_layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer1/bias/Regularizer/ConstО
dnn_layer1/bias/Regularizer/SumSum&dnn_layer1/bias/Regularizer/Square:y:0*dnn_layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/Sum
!dnn_layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer1/bias/Regularizer/mul/xР
dnn_layer1/bias/Regularizer/mulMul*dnn_layer1/bias/Regularizer/mul/x:output:0(dnn_layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/mulН
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer2_75433* 
_output_shapes
:
*
dtype025
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer2/kernel/Regularizer/SquareSquare;dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer2/kernel/Regularizer/Square
#dnn_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer2/kernel/Regularizer/ConstЦ
!dnn_layer2/kernel/Regularizer/SumSum(dnn_layer2/kernel/Regularizer/Square:y:0,dnn_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/Sum
#dnn_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer2/kernel/Regularizer/mul/xШ
!dnn_layer2/kernel/Regularizer/mulMul,dnn_layer2/kernel/Regularizer/mul/x:output:0*dnn_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/mulД
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer2_75435*
_output_shapes	
:*
dtype023
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer2/bias/Regularizer/SquareSquare9dnn_layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer2/bias/Regularizer/Square
!dnn_layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer2/bias/Regularizer/ConstО
dnn_layer2/bias/Regularizer/SumSum&dnn_layer2/bias/Regularizer/Square:y:0*dnn_layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/Sum
!dnn_layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer2/bias/Regularizer/mul/xР
dnn_layer2/bias/Regularizer/mulMul*dnn_layer2/bias/Regularizer/mul/x:output:0(dnn_layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/mulН
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer3_75439* 
_output_shapes
:
*
dtype025
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer3/kernel/Regularizer/SquareSquare;dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer3/kernel/Regularizer/Square
#dnn_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer3/kernel/Regularizer/ConstЦ
!dnn_layer3/kernel/Regularizer/SumSum(dnn_layer3/kernel/Regularizer/Square:y:0,dnn_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/Sum
#dnn_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer3/kernel/Regularizer/mul/xШ
!dnn_layer3/kernel/Regularizer/mulMul,dnn_layer3/kernel/Regularizer/mul/x:output:0*dnn_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/mulД
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer3_75441*
_output_shapes	
:*
dtype023
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer3/bias/Regularizer/SquareSquare9dnn_layer3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer3/bias/Regularizer/Square
!dnn_layer3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer3/bias/Regularizer/ConstО
dnn_layer3/bias/Regularizer/SumSum&dnn_layer3/bias/Regularizer/Square:y:0*dnn_layer3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/Sum
!dnn_layer3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer3/bias/Regularizer/mul/xР
dnn_layer3/bias/Regularizer/mulMul*dnn_layer3/bias/Regularizer/mul/x:output:0(dnn_layer3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/mulЙ
2fm_linear/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfm_linear_75446*
_output_shapes
:	*
dtype024
2fm_linear/kernel/Regularizer/Square/ReadVariableOpК
#fm_linear/kernel/Regularizer/SquareSquare:fm_linear/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#fm_linear/kernel/Regularizer/Square
"fm_linear/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"fm_linear/kernel/Regularizer/ConstТ
 fm_linear/kernel/Regularizer/SumSum'fm_linear/kernel/Regularizer/Square:y:0+fm_linear/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/Sum
"fm_linear/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2$
"fm_linear/kernel/Regularizer/mul/xФ
 fm_linear/kernel/Regularizer/mulMul+fm_linear/kernel/Regularizer/mul/x:output:0)fm_linear/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/mulА
0fm_linear/bias/Regularizer/Square/ReadVariableOpReadVariableOpfm_linear_75448*
_output_shapes
:*
dtype022
0fm_linear/bias/Regularizer/Square/ReadVariableOpЏ
!fm_linear/bias/Regularizer/SquareSquare8fm_linear/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!fm_linear/bias/Regularizer/Square
 fm_linear/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 fm_linear/bias/Regularizer/ConstК
fm_linear/bias/Regularizer/SumSum%fm_linear/bias/Regularizer/Square:y:0)fm_linear/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/Sum
 fm_linear/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2"
 fm_linear/bias/Regularizer/mul/xМ
fm_linear/bias/Regularizer/mulMul)fm_linear/bias/Regularizer/mul/x:output:0'fm_linear/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/mulМ
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer4_75453*
_output_shapes
:	*
dtype025
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpН
$dnn_layer4/kernel/Regularizer/SquareSquare;dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$dnn_layer4/kernel/Regularizer/Square
#dnn_layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer4/kernel/Regularizer/ConstЦ
!dnn_layer4/kernel/Regularizer/SumSum(dnn_layer4/kernel/Regularizer/Square:y:0,dnn_layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/Sum
#dnn_layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer4/kernel/Regularizer/mul/xШ
!dnn_layer4/kernel/Regularizer/mulMul,dnn_layer4/kernel/Regularizer/mul/x:output:0*dnn_layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/mul
IdentityIdentity sigmoid/PartitionedCall:output:0#^dnn_layer1/StatefulPartitionedCall2^dnn_layer1/bias/Regularizer/Square/ReadVariableOp4^dnn_layer1/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer2/StatefulPartitionedCall2^dnn_layer2/bias/Regularizer/Square/ReadVariableOp4^dnn_layer2/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer3/StatefulPartitionedCall2^dnn_layer3/bias/Regularizer/Square/ReadVariableOp4^dnn_layer3/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer4/StatefulPartitionedCall4^dnn_layer4/kernel/Regularizer/Square/ReadVariableOp"^fm_linear/StatefulPartitionedCall1^fm_linear/bias/Regularizer/Square/ReadVariableOp3^fm_linear/kernel/Regularizer/Square/ReadVariableOp/^item_catalog_embedding/StatefulPartitionedCall*^item_id_embedding/StatefulPartitionedCall+^item_tag_embedding/StatefulPartitionedCall.^member_type_embedding/StatefulPartitionedCall*^user_id_embedding/StatefulPartitionedCall,^user_type_embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::2H
"dnn_layer1/StatefulPartitionedCall"dnn_layer1/StatefulPartitionedCall2f
1dnn_layer1/bias/Regularizer/Square/ReadVariableOp1dnn_layer1/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer2/StatefulPartitionedCall"dnn_layer2/StatefulPartitionedCall2f
1dnn_layer2/bias/Regularizer/Square/ReadVariableOp1dnn_layer2/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer3/StatefulPartitionedCall"dnn_layer3/StatefulPartitionedCall2f
1dnn_layer3/bias/Regularizer/Square/ReadVariableOp1dnn_layer3/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer4/StatefulPartitionedCall"dnn_layer4/StatefulPartitionedCall2j
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp2F
!fm_linear/StatefulPartitionedCall!fm_linear/StatefulPartitionedCall2d
0fm_linear/bias/Regularizer/Square/ReadVariableOp0fm_linear/bias/Regularizer/Square/ReadVariableOp2h
2fm_linear/kernel/Regularizer/Square/ReadVariableOp2fm_linear/kernel/Regularizer/Square/ReadVariableOp2`
.item_catalog_embedding/StatefulPartitionedCall.item_catalog_embedding/StatefulPartitionedCall2V
)item_id_embedding/StatefulPartitionedCall)item_id_embedding/StatefulPartitionedCall2X
*item_tag_embedding/StatefulPartitionedCall*item_tag_embedding/StatefulPartitionedCall2^
-member_type_embedding/StatefulPartitionedCall-member_type_embedding/StatefulPartitionedCall2V
)user_id_embedding/StatefulPartitionedCall)user_id_embedding/StatefulPartitionedCall2Z
+user_type_embedding/StatefulPartitionedCall+user_type_embedding/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
п
І
__inference_loss_fn_0_76529@
<dnn_layer1_kernel_regularizer_square_readvariableop_resource
identityЂ3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpщ
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<dnn_layer1_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer1/kernel/Regularizer/SquareSquare;dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer1/kernel/Regularizer/Square
#dnn_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer1/kernel/Regularizer/ConstЦ
!dnn_layer1/kernel/Regularizer/SumSum(dnn_layer1/kernel/Regularizer/Square:y:0,dnn_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/Sum
#dnn_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer1/kernel/Regularizer/mul/xШ
!dnn_layer1/kernel/Regularizer/mulMul,dnn_layer1/kernel/Regularizer/mul/x:output:0*dnn_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/mul
IdentityIdentity%dnn_layer1/kernel/Regularizer/mul:z:04^dnn_layer1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2j
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp

Ђ
__inference_loss_fn_5_76584>
:dnn_layer3_bias_regularizer_square_readvariableop_resource
identityЂ1dnn_layer3/bias/Regularizer/Square/ReadVariableOpо
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpReadVariableOp:dnn_layer3_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer3/bias/Regularizer/SquareSquare9dnn_layer3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer3/bias/Regularizer/Square
!dnn_layer3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer3/bias/Regularizer/ConstО
dnn_layer3/bias/Regularizer/SumSum&dnn_layer3/bias/Regularizer/Square:y:0*dnn_layer3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/Sum
!dnn_layer3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer3/bias/Regularizer/mul/xР
dnn_layer3/bias/Regularizer/mulMul*dnn_layer3/bias/Regularizer/mul/x:output:0(dnn_layer3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/mul
IdentityIdentity#dnn_layer3/bias/Regularizer/mul:z:02^dnn_layer3/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dnn_layer3/bias/Regularizer/Square/ReadVariableOp1dnn_layer3/bias/Regularizer/Square/ReadVariableOp
Ѓ
o
E__inference_fm_combine_layer_call_and_return_conditional_losses_75006

inputs
inputs_1
identityW
addAddV2inputsinputs_1*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
a
E__inference_flatten_16_layer_call_and_return_conditional_losses_74901

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	

N__inference_user_type_embedding_layer_call_and_return_conditional_losses_74605

inputs
embedding_lookup_74599
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_74599inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/74599*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/74599*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	

Q__inference_item_catalog_embedding_layer_call_and_return_conditional_losses_74647

inputs
embedding_lookup_74641
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_74641inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/74641*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/74641*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ж
E__inference_dnn_layer4_layer_call_and_return_conditional_losses_76489

inputs"
matmul_readvariableop_resource
identityЂMatMul/ReadVariableOpЂ3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMulЪ
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype025
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpН
$dnn_layer4/kernel/Regularizer/SquareSquare;dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$dnn_layer4/kernel/Regularizer/Square
#dnn_layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer4/kernel/Regularizer/ConstЦ
!dnn_layer4/kernel/Regularizer/SumSum(dnn_layer4/kernel/Regularizer/Square:y:0,dnn_layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/Sum
#dnn_layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer4/kernel/Regularizer/mul/xШ
!dnn_layer4/kernel/Regularizer/mulMul,dnn_layer4/kernel/Regularizer/mul/x:output:0*dnn_layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/mulВ
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp4^dnn_layer4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х

*__inference_dnn_layer1_layer_call_fn_76215

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer1_layer_call_and_return_conditional_losses_747412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	

M__inference_item_tag_embedding_layer_call_and_return_conditional_losses_74668

inputs
embedding_lookup_74662
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_74662inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/74662*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/74662*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


(__inference_model_17_layer_call_fn_76043
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_17_layer_call_and_return_conditional_losses_755132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/5

Ђ
__inference_loss_fn_1_76540>
:dnn_layer1_bias_regularizer_square_readvariableop_resource
identityЂ1dnn_layer1/bias/Regularizer/Square/ReadVariableOpо
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOp:dnn_layer1_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer1/bias/Regularizer/SquareSquare9dnn_layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer1/bias/Regularizer/Square
!dnn_layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer1/bias/Regularizer/ConstО
dnn_layer1/bias/Regularizer/SumSum&dnn_layer1/bias/Regularizer/Square:y:0*dnn_layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/Sum
!dnn_layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer1/bias/Regularizer/mul/xР
dnn_layer1/bias/Regularizer/mulMul*dnn_layer1/bias/Regularizer/mul/x:output:0(dnn_layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/mul
IdentityIdentity#dnn_layer1/bias/Regularizer/mul:z:02^dnn_layer1/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dnn_layer1/bias/Regularizer/Square/ReadVariableOp1dnn_layer1/bias/Regularizer/Square/ReadVariableOp
Ь
c
E__inference_dropout_25_layer_call_and_return_conditional_losses_74843

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
E__inference_dropout_24_layer_call_and_return_conditional_losses_76227

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с
~
)__inference_fm_linear_layer_call_fn_76411

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_fm_linear_layer_call_and_return_conditional_losses_749612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л
і
C__inference_model_17_layer_call_and_return_conditional_losses_75963
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5,
(user_id_embedding_embedding_lookup_758250
,member_type_embedding_embedding_lookup_75830.
*user_type_embedding_embedding_lookup_75835,
(item_id_embedding_embedding_lookup_758401
-item_catalog_embedding_embedding_lookup_75845-
)item_tag_embedding_embedding_lookup_75850-
)dnn_layer1_matmul_readvariableop_resource.
*dnn_layer1_biasadd_readvariableop_resource-
)dnn_layer2_matmul_readvariableop_resource.
*dnn_layer2_biasadd_readvariableop_resource-
)dnn_layer3_matmul_readvariableop_resource.
*dnn_layer3_biasadd_readvariableop_resource,
(fm_linear_matmul_readvariableop_resource-
)fm_linear_biasadd_readvariableop_resource-
)dnn_layer4_matmul_readvariableop_resource
identityЂ!dnn_layer1/BiasAdd/ReadVariableOpЂ dnn_layer1/MatMul/ReadVariableOpЂ1dnn_layer1/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpЂ!dnn_layer2/BiasAdd/ReadVariableOpЂ dnn_layer2/MatMul/ReadVariableOpЂ1dnn_layer2/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpЂ!dnn_layer3/BiasAdd/ReadVariableOpЂ dnn_layer3/MatMul/ReadVariableOpЂ1dnn_layer3/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpЂ dnn_layer4/MatMul/ReadVariableOpЂ3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpЂ fm_linear/BiasAdd/ReadVariableOpЂfm_linear/MatMul/ReadVariableOpЂ0fm_linear/bias/Regularizer/Square/ReadVariableOpЂ2fm_linear/kernel/Regularizer/Square/ReadVariableOpЂ'item_catalog_embedding/embedding_lookupЂ"item_id_embedding/embedding_lookupЂ#item_tag_embedding/embedding_lookupЂ&member_type_embedding/embedding_lookupЂ"user_id_embedding/embedding_lookupЂ$user_type_embedding/embedding_lookupУ
"user_id_embedding/embedding_lookupResourceGather(user_id_embedding_embedding_lookup_75825inputs_0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@user_id_embedding/embedding_lookup/75825*+
_output_shapes
:џџџџџџџџџ*
dtype02$
"user_id_embedding/embedding_lookupД
+user_id_embedding/embedding_lookup/IdentityIdentity+user_id_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@user_id_embedding/embedding_lookup/75825*+
_output_shapes
:џџџџџџџџџ2-
+user_id_embedding/embedding_lookup/Identityж
-user_id_embedding/embedding_lookup/Identity_1Identity4user_id_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2/
-user_id_embedding/embedding_lookup/Identity_1г
&member_type_embedding/embedding_lookupResourceGather,member_type_embedding_embedding_lookup_75830inputs_2",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*?
_class5
31loc:@member_type_embedding/embedding_lookup/75830*+
_output_shapes
:џџџџџџџџџ*
dtype02(
&member_type_embedding/embedding_lookupФ
/member_type_embedding/embedding_lookup/IdentityIdentity/member_type_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@member_type_embedding/embedding_lookup/75830*+
_output_shapes
:џџџџџџџџџ21
/member_type_embedding/embedding_lookup/Identityт
1member_type_embedding/embedding_lookup/Identity_1Identity8member_type_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ23
1member_type_embedding/embedding_lookup/Identity_1Ы
$user_type_embedding/embedding_lookupResourceGather*user_type_embedding_embedding_lookup_75835inputs_1",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@user_type_embedding/embedding_lookup/75835*+
_output_shapes
:џџџџџџџџџ*
dtype02&
$user_type_embedding/embedding_lookupМ
-user_type_embedding/embedding_lookup/IdentityIdentity-user_type_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@user_type_embedding/embedding_lookup/75835*+
_output_shapes
:џџџџџџџџџ2/
-user_type_embedding/embedding_lookup/Identityм
/user_type_embedding/embedding_lookup/Identity_1Identity6user_type_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ21
/user_type_embedding/embedding_lookup/Identity_1У
"item_id_embedding/embedding_lookupResourceGather(item_id_embedding_embedding_lookup_75840inputs_3",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@item_id_embedding/embedding_lookup/75840*+
_output_shapes
:џџџџџџџџџ*
dtype02$
"item_id_embedding/embedding_lookupД
+item_id_embedding/embedding_lookup/IdentityIdentity+item_id_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@item_id_embedding/embedding_lookup/75840*+
_output_shapes
:џџџџџџџџџ2-
+item_id_embedding/embedding_lookup/Identityж
-item_id_embedding/embedding_lookup/Identity_1Identity4item_id_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2/
-item_id_embedding/embedding_lookup/Identity_1з
'item_catalog_embedding/embedding_lookupResourceGather-item_catalog_embedding_embedding_lookup_75845inputs_4",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*@
_class6
42loc:@item_catalog_embedding/embedding_lookup/75845*+
_output_shapes
:џџџџџџџџџ*
dtype02)
'item_catalog_embedding/embedding_lookupШ
0item_catalog_embedding/embedding_lookup/IdentityIdentity0item_catalog_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@item_catalog_embedding/embedding_lookup/75845*+
_output_shapes
:џџџџџџџџџ22
0item_catalog_embedding/embedding_lookup/Identityх
2item_catalog_embedding/embedding_lookup/Identity_1Identity9item_catalog_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ24
2item_catalog_embedding/embedding_lookup/Identity_1Ч
#item_tag_embedding/embedding_lookupResourceGather)item_tag_embedding_embedding_lookup_75850inputs_5",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@item_tag_embedding/embedding_lookup/75850*+
_output_shapes
:џџџџџџџџџ*
dtype02%
#item_tag_embedding/embedding_lookupИ
,item_tag_embedding/embedding_lookup/IdentityIdentity,item_tag_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@item_tag_embedding/embedding_lookup/75850*+
_output_shapes
:џџџџџџџџџ2.
,item_tag_embedding/embedding_lookup/Identityй
.item_tag_embedding/embedding_lookup/Identity_1Identity5item_tag_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ20
.item_tag_embedding/embedding_lookup/Identity_1
!embedding_concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!embedding_concatenate/concat/axis
embedding_concatenate/concatConcatV26user_id_embedding/embedding_lookup/Identity_1:output:0:member_type_embedding/embedding_lookup/Identity_1:output:08user_type_embedding/embedding_lookup/Identity_1:output:06item_id_embedding/embedding_lookup/Identity_1:output:0;item_catalog_embedding/embedding_lookup/Identity_1:output:07item_tag_embedding/embedding_lookup/Identity_1:output:0*embedding_concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_concatenate/concatu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_17/ConstЈ
flatten_17/ReshapeReshape%embedding_concatenate/concat:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_17/ReshapeА
 dnn_layer1/MatMul/ReadVariableOpReadVariableOp)dnn_layer1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dnn_layer1/MatMul/ReadVariableOpЊ
dnn_layer1/MatMulMatMulflatten_17/Reshape:output:0(dnn_layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer1/MatMulЎ
!dnn_layer1/BiasAdd/ReadVariableOpReadVariableOp*dnn_layer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dnn_layer1/BiasAdd/ReadVariableOpЎ
dnn_layer1/BiasAddBiasAdddnn_layer1/MatMul:product:0)dnn_layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer1/BiasAddz
dnn_layer1/ReluReludnn_layer1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer1/Relu
dropout_24/IdentityIdentitydnn_layer1/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_24/IdentityА
 dnn_layer2/MatMul/ReadVariableOpReadVariableOp)dnn_layer2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dnn_layer2/MatMul/ReadVariableOpЋ
dnn_layer2/MatMulMatMuldropout_24/Identity:output:0(dnn_layer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer2/MatMulЎ
!dnn_layer2/BiasAdd/ReadVariableOpReadVariableOp*dnn_layer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dnn_layer2/BiasAdd/ReadVariableOpЎ
dnn_layer2/BiasAddBiasAdddnn_layer2/MatMul:product:0)dnn_layer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer2/BiasAddz
dnn_layer2/ReluReludnn_layer2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer2/Relu
dropout_25/IdentityIdentitydnn_layer2/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_25/IdentityА
 dnn_layer3/MatMul/ReadVariableOpReadVariableOp)dnn_layer3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dnn_layer3/MatMul/ReadVariableOpЋ
dnn_layer3/MatMulMatMuldropout_25/Identity:output:0(dnn_layer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer3/MatMulЎ
!dnn_layer3/BiasAdd/ReadVariableOpReadVariableOp*dnn_layer3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dnn_layer3/BiasAdd/ReadVariableOpЎ
dnn_layer3/BiasAddBiasAdddnn_layer3/MatMul:product:0)dnn_layer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer3/BiasAddz
dnn_layer3/ReluReludnn_layer3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer3/Reluu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_16/ConstЈ
flatten_16/ReshapeReshape%embedding_concatenate/concat:output:0flatten_16/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_16/Reshape
dropout_26/IdentityIdentitydnn_layer3/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_26/IdentityЌ
fm_linear/MatMul/ReadVariableOpReadVariableOp(fm_linear_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
fm_linear/MatMul/ReadVariableOpІ
fm_linear/MatMulMatMulflatten_16/Reshape:output:0'fm_linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_linear/MatMulЊ
 fm_linear/BiasAdd/ReadVariableOpReadVariableOp)fm_linear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 fm_linear/BiasAdd/ReadVariableOpЉ
fm_linear/BiasAddBiasAddfm_linear/MatMul:product:0(fm_linear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_linear/BiasAdd
fm_cross/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
fm_cross/Sum/reduction_indicesК
fm_cross/SumSum%embedding_concatenate/concat:output:0'fm_cross/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
fm_cross/Sumy
fm_cross/SquareSquarefm_cross/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
fm_cross/Square
fm_cross/Square_1Square%embedding_concatenate/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
fm_cross/Square_1
 fm_cross/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 fm_cross/Sum_1/reduction_indicesА
fm_cross/Sum_1Sumfm_cross/Square_1:y:0)fm_cross/Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
fm_cross/Sum_1
fm_cross/subSubfm_cross/Square:y:0fm_cross/Sum_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
fm_cross/sub
 fm_cross/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 fm_cross/Sum_2/reduction_indices
fm_cross/Sum_2Sumfm_cross/sub:z:0)fm_cross/Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_cross/Sum_2e
fm_cross/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
fm_cross/mul/x
fm_cross/mulMulfm_cross/mul/x:output:0fm_cross/Sum_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_cross/mul
fm_combine/addAddV2fm_linear/BiasAdd:output:0fm_cross/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_combine/addЏ
 dnn_layer4/MatMul/ReadVariableOpReadVariableOp)dnn_layer4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dnn_layer4/MatMul/ReadVariableOpЊ
dnn_layer4/MatMulMatMuldropout_26/Identity:output:0(dnn_layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dnn_layer4/MatMul
	add_8/addAddV2fm_combine/add:z:0dnn_layer4/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	add_8/addn
sigmoid/SigmoidSigmoidadd_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sigmoid/Sigmoidж
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dnn_layer1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer1/kernel/Regularizer/SquareSquare;dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer1/kernel/Regularizer/Square
#dnn_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer1/kernel/Regularizer/ConstЦ
!dnn_layer1/kernel/Regularizer/SumSum(dnn_layer1/kernel/Regularizer/Square:y:0,dnn_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/Sum
#dnn_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer1/kernel/Regularizer/mul/xШ
!dnn_layer1/kernel/Regularizer/mulMul,dnn_layer1/kernel/Regularizer/mul/x:output:0*dnn_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/mulЮ
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOp*dnn_layer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer1/bias/Regularizer/SquareSquare9dnn_layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer1/bias/Regularizer/Square
!dnn_layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer1/bias/Regularizer/ConstО
dnn_layer1/bias/Regularizer/SumSum&dnn_layer1/bias/Regularizer/Square:y:0*dnn_layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/Sum
!dnn_layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer1/bias/Regularizer/mul/xР
dnn_layer1/bias/Regularizer/mulMul*dnn_layer1/bias/Regularizer/mul/x:output:0(dnn_layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/mulж
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dnn_layer2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer2/kernel/Regularizer/SquareSquare;dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer2/kernel/Regularizer/Square
#dnn_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer2/kernel/Regularizer/ConstЦ
!dnn_layer2/kernel/Regularizer/SumSum(dnn_layer2/kernel/Regularizer/Square:y:0,dnn_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/Sum
#dnn_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer2/kernel/Regularizer/mul/xШ
!dnn_layer2/kernel/Regularizer/mulMul,dnn_layer2/kernel/Regularizer/mul/x:output:0*dnn_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/mulЮ
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOp*dnn_layer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer2/bias/Regularizer/SquareSquare9dnn_layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer2/bias/Regularizer/Square
!dnn_layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer2/bias/Regularizer/ConstО
dnn_layer2/bias/Regularizer/SumSum&dnn_layer2/bias/Regularizer/Square:y:0*dnn_layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/Sum
!dnn_layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer2/bias/Regularizer/mul/xР
dnn_layer2/bias/Regularizer/mulMul*dnn_layer2/bias/Regularizer/mul/x:output:0(dnn_layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/mulж
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dnn_layer3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer3/kernel/Regularizer/SquareSquare;dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer3/kernel/Regularizer/Square
#dnn_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer3/kernel/Regularizer/ConstЦ
!dnn_layer3/kernel/Regularizer/SumSum(dnn_layer3/kernel/Regularizer/Square:y:0,dnn_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/Sum
#dnn_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer3/kernel/Regularizer/mul/xШ
!dnn_layer3/kernel/Regularizer/mulMul,dnn_layer3/kernel/Regularizer/mul/x:output:0*dnn_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/mulЮ
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpReadVariableOp*dnn_layer3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer3/bias/Regularizer/SquareSquare9dnn_layer3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer3/bias/Regularizer/Square
!dnn_layer3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer3/bias/Regularizer/ConstО
dnn_layer3/bias/Regularizer/SumSum&dnn_layer3/bias/Regularizer/Square:y:0*dnn_layer3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/Sum
!dnn_layer3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer3/bias/Regularizer/mul/xР
dnn_layer3/bias/Regularizer/mulMul*dnn_layer3/bias/Regularizer/mul/x:output:0(dnn_layer3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/mulв
2fm_linear/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(fm_linear_matmul_readvariableop_resource*
_output_shapes
:	*
dtype024
2fm_linear/kernel/Regularizer/Square/ReadVariableOpК
#fm_linear/kernel/Regularizer/SquareSquare:fm_linear/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#fm_linear/kernel/Regularizer/Square
"fm_linear/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"fm_linear/kernel/Regularizer/ConstТ
 fm_linear/kernel/Regularizer/SumSum'fm_linear/kernel/Regularizer/Square:y:0+fm_linear/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/Sum
"fm_linear/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2$
"fm_linear/kernel/Regularizer/mul/xФ
 fm_linear/kernel/Regularizer/mulMul+fm_linear/kernel/Regularizer/mul/x:output:0)fm_linear/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/mulЪ
0fm_linear/bias/Regularizer/Square/ReadVariableOpReadVariableOp)fm_linear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0fm_linear/bias/Regularizer/Square/ReadVariableOpЏ
!fm_linear/bias/Regularizer/SquareSquare8fm_linear/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!fm_linear/bias/Regularizer/Square
 fm_linear/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 fm_linear/bias/Regularizer/ConstК
fm_linear/bias/Regularizer/SumSum%fm_linear/bias/Regularizer/Square:y:0)fm_linear/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/Sum
 fm_linear/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2"
 fm_linear/bias/Regularizer/mul/xМ
fm_linear/bias/Regularizer/mulMul)fm_linear/bias/Regularizer/mul/x:output:0'fm_linear/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/mulе
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dnn_layer4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype025
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpН
$dnn_layer4/kernel/Regularizer/SquareSquare;dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$dnn_layer4/kernel/Regularizer/Square
#dnn_layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer4/kernel/Regularizer/ConstЦ
!dnn_layer4/kernel/Regularizer/SumSum(dnn_layer4/kernel/Regularizer/Square:y:0,dnn_layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/Sum
#dnn_layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer4/kernel/Regularizer/mul/xШ
!dnn_layer4/kernel/Regularizer/mulMul,dnn_layer4/kernel/Regularizer/mul/x:output:0*dnn_layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/mulъ
IdentityIdentitysigmoid/Sigmoid:y:0"^dnn_layer1/BiasAdd/ReadVariableOp!^dnn_layer1/MatMul/ReadVariableOp2^dnn_layer1/bias/Regularizer/Square/ReadVariableOp4^dnn_layer1/kernel/Regularizer/Square/ReadVariableOp"^dnn_layer2/BiasAdd/ReadVariableOp!^dnn_layer2/MatMul/ReadVariableOp2^dnn_layer2/bias/Regularizer/Square/ReadVariableOp4^dnn_layer2/kernel/Regularizer/Square/ReadVariableOp"^dnn_layer3/BiasAdd/ReadVariableOp!^dnn_layer3/MatMul/ReadVariableOp2^dnn_layer3/bias/Regularizer/Square/ReadVariableOp4^dnn_layer3/kernel/Regularizer/Square/ReadVariableOp!^dnn_layer4/MatMul/ReadVariableOp4^dnn_layer4/kernel/Regularizer/Square/ReadVariableOp!^fm_linear/BiasAdd/ReadVariableOp ^fm_linear/MatMul/ReadVariableOp1^fm_linear/bias/Regularizer/Square/ReadVariableOp3^fm_linear/kernel/Regularizer/Square/ReadVariableOp(^item_catalog_embedding/embedding_lookup#^item_id_embedding/embedding_lookup$^item_tag_embedding/embedding_lookup'^member_type_embedding/embedding_lookup#^user_id_embedding/embedding_lookup%^user_type_embedding/embedding_lookup*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::2F
!dnn_layer1/BiasAdd/ReadVariableOp!dnn_layer1/BiasAdd/ReadVariableOp2D
 dnn_layer1/MatMul/ReadVariableOp dnn_layer1/MatMul/ReadVariableOp2f
1dnn_layer1/bias/Regularizer/Square/ReadVariableOp1dnn_layer1/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp2F
!dnn_layer2/BiasAdd/ReadVariableOp!dnn_layer2/BiasAdd/ReadVariableOp2D
 dnn_layer2/MatMul/ReadVariableOp dnn_layer2/MatMul/ReadVariableOp2f
1dnn_layer2/bias/Regularizer/Square/ReadVariableOp1dnn_layer2/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp2F
!dnn_layer3/BiasAdd/ReadVariableOp!dnn_layer3/BiasAdd/ReadVariableOp2D
 dnn_layer3/MatMul/ReadVariableOp dnn_layer3/MatMul/ReadVariableOp2f
1dnn_layer3/bias/Regularizer/Square/ReadVariableOp1dnn_layer3/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp2D
 dnn_layer4/MatMul/ReadVariableOp dnn_layer4/MatMul/ReadVariableOp2j
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp2D
 fm_linear/BiasAdd/ReadVariableOp fm_linear/BiasAdd/ReadVariableOp2B
fm_linear/MatMul/ReadVariableOpfm_linear/MatMul/ReadVariableOp2d
0fm_linear/bias/Regularizer/Square/ReadVariableOp0fm_linear/bias/Regularizer/Square/ReadVariableOp2h
2fm_linear/kernel/Regularizer/Square/ReadVariableOp2fm_linear/kernel/Regularizer/Square/ReadVariableOp2R
'item_catalog_embedding/embedding_lookup'item_catalog_embedding/embedding_lookup2H
"item_id_embedding/embedding_lookup"item_id_embedding/embedding_lookup2J
#item_tag_embedding/embedding_lookup#item_tag_embedding/embedding_lookup2P
&member_type_embedding/embedding_lookup&member_type_embedding/embedding_lookup2H
"user_id_embedding/embedding_lookup"user_id_embedding/embedding_lookup2L
$user_type_embedding/embedding_lookup$user_type_embedding/embedding_lookup:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/5
џ

L__inference_item_id_embedding_layer_call_and_return_conditional_losses_76100

inputs
embedding_lookup_76094
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_76094inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/76094*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/76094*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
F
*__inference_flatten_16_layer_call_fn_76324

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_749012
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

V
*__inference_fm_combine_layer_call_fn_76470
inputs_0
inputs_1
identityг
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fm_combine_layer_call_and_return_conditional_losses_750062
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1

ж
E__inference_dnn_layer4_layer_call_and_return_conditional_losses_75028

inputs"
matmul_readvariableop_resource
identityЂMatMul/ReadVariableOpЂ3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMulЪ
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype025
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpН
$dnn_layer4/kernel/Regularizer/SquareSquare;dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$dnn_layer4/kernel/Regularizer/Square
#dnn_layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer4/kernel/Regularizer/ConstЦ
!dnn_layer4/kernel/Regularizer/SumSum(dnn_layer4/kernel/Regularizer/Square:y:0,dnn_layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/Sum
#dnn_layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer4/kernel/Regularizer/mul/xШ
!dnn_layer4/kernel/Regularizer/mulMul,dnn_layer4/kernel/Regularizer/mul/x:output:0*dnn_layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/mulВ
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp4^dnn_layer4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ

L__inference_user_id_embedding_layer_call_and_return_conditional_losses_76052

inputs
embedding_lookup_76046
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_76046inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/76046*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/76046*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
c
*__inference_dropout_25_layer_call_fn_76308

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_748382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	

M__inference_item_tag_embedding_layer_call_and_return_conditional_losses_76132

inputs
embedding_lookup_76126
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_76126inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/76126*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/76126*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ш
E__inference_dnn_layer3_layer_call_and_return_conditional_losses_76359

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ1dnn_layer3/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluЫ
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer3/kernel/Regularizer/SquareSquare;dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer3/kernel/Regularizer/Square
#dnn_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer3/kernel/Regularizer/ConstЦ
!dnn_layer3/kernel/Regularizer/SumSum(dnn_layer3/kernel/Regularizer/Square:y:0,dnn_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/Sum
#dnn_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer3/kernel/Regularizer/mul/xШ
!dnn_layer3/kernel/Regularizer/mulMul,dnn_layer3/kernel/Regularizer/mul/x:output:0*dnn_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/mulУ
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer3/bias/Regularizer/SquareSquare9dnn_layer3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer3/bias/Regularizer/Square
!dnn_layer3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer3/bias/Regularizer/ConstО
dnn_layer3/bias/Regularizer/SumSum&dnn_layer3/bias/Regularizer/Square:y:0*dnn_layer3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/Sum
!dnn_layer3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer3/bias/Regularizer/mul/xР
dnn_layer3/bias/Regularizer/mulMul*dnn_layer3/bias/Regularizer/mul/x:output:0(dnn_layer3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/mul
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dnn_layer3/bias/Regularizer/Square/ReadVariableOp4^dnn_layer3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dnn_layer3/bias/Regularizer/Square/ReadVariableOp1dnn_layer3/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
q
E__inference_fm_combine_layer_call_and_return_conditional_losses_76464
inputs_0
inputs_1
identityY
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
є

#__inference_signature_wrapper_75650
item_catalog
item_id
item_tag
member_type
user_id
	user_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCalluser_id	user_typemember_typeitem_iditem_catalogitem_tagunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_745452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameitem_catalog:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	item_id:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
item_tag:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_namemember_type:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	user_id:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	user_type


(__inference_model_17_layer_call_fn_75546
user_id
	user_type
member_type
item_id
item_catalog
item_tag
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCalluser_id	user_typemember_typeitem_iditem_catalogitem_tagunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_17_layer_call_and_return_conditional_losses_755132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	user_id:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	user_type:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_namemember_type:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	item_id:UQ
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameitem_catalog:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
item_tag

Ш
E__inference_dnn_layer3_layer_call_and_return_conditional_losses_74879

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ1dnn_layer3/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluЫ
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer3/kernel/Regularizer/SquareSquare;dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer3/kernel/Regularizer/Square
#dnn_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer3/kernel/Regularizer/ConstЦ
!dnn_layer3/kernel/Regularizer/SumSum(dnn_layer3/kernel/Regularizer/Square:y:0,dnn_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/Sum
#dnn_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer3/kernel/Regularizer/mul/xШ
!dnn_layer3/kernel/Regularizer/mulMul,dnn_layer3/kernel/Regularizer/mul/x:output:0*dnn_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/mulУ
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer3/bias/Regularizer/SquareSquare9dnn_layer3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer3/bias/Regularizer/Square
!dnn_layer3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer3/bias/Regularizer/ConstО
dnn_layer3/bias/Regularizer/SumSum&dnn_layer3/bias/Regularizer/Square:y:0*dnn_layer3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/Sum
!dnn_layer3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer3/bias/Regularizer/mul/xР
dnn_layer3/bias/Regularizer/mulMul*dnn_layer3/bias/Regularizer/mul/x:output:0(dnn_layer3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/mul
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dnn_layer3/bias/Regularizer/Square/ReadVariableOp4^dnn_layer3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dnn_layer3/bias/Regularizer/Square/ReadVariableOp1dnn_layer3/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ

L__inference_item_id_embedding_layer_call_and_return_conditional_losses_74626

inputs
embedding_lookup_74620
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_74620inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/74620*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/74620*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ	
В
P__inference_embedding_concatenate_layer_call_and_return_conditional_losses_74691

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЋ
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:SO
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:SO
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:SO
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:SO
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:SO
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І
l
@__inference_add_8_layer_call_and_return_conditional_losses_76502
inputs_0
inputs_1
identityY
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
ЁК
ж
C__inference_model_17_layer_call_and_return_conditional_losses_75237
user_id
	user_type
member_type
item_id
item_catalog
item_tag
user_id_embedding_75131
member_type_embedding_75134
user_type_embedding_75137
item_id_embedding_75140 
item_catalog_embedding_75143
item_tag_embedding_75146
dnn_layer1_75151
dnn_layer1_75153
dnn_layer2_75157
dnn_layer2_75159
dnn_layer3_75163
dnn_layer3_75165
fm_linear_75170
fm_linear_75172
dnn_layer4_75177
identityЂ"dnn_layer1/StatefulPartitionedCallЂ1dnn_layer1/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer2/StatefulPartitionedCallЂ1dnn_layer2/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer3/StatefulPartitionedCallЂ1dnn_layer3/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer4/StatefulPartitionedCallЂ3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpЂ!fm_linear/StatefulPartitionedCallЂ0fm_linear/bias/Regularizer/Square/ReadVariableOpЂ2fm_linear/kernel/Regularizer/Square/ReadVariableOpЂ.item_catalog_embedding/StatefulPartitionedCallЂ)item_id_embedding/StatefulPartitionedCallЂ*item_tag_embedding/StatefulPartitionedCallЂ-member_type_embedding/StatefulPartitionedCallЂ)user_id_embedding/StatefulPartitionedCallЂ+user_type_embedding/StatefulPartitionedCallЋ
)user_id_embedding/StatefulPartitionedCallStatefulPartitionedCalluser_iduser_id_embedding_75131*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_user_id_embedding_layer_call_and_return_conditional_losses_745632+
)user_id_embedding/StatefulPartitionedCallП
-member_type_embedding/StatefulPartitionedCallStatefulPartitionedCallmember_typemember_type_embedding_75134*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_member_type_embedding_layer_call_and_return_conditional_losses_745842/
-member_type_embedding/StatefulPartitionedCallЕ
+user_type_embedding/StatefulPartitionedCallStatefulPartitionedCall	user_typeuser_type_embedding_75137*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_user_type_embedding_layer_call_and_return_conditional_losses_746052-
+user_type_embedding/StatefulPartitionedCallЋ
)item_id_embedding/StatefulPartitionedCallStatefulPartitionedCallitem_iditem_id_embedding_75140*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_item_id_embedding_layer_call_and_return_conditional_losses_746262+
)item_id_embedding/StatefulPartitionedCallФ
.item_catalog_embedding/StatefulPartitionedCallStatefulPartitionedCallitem_catalogitem_catalog_embedding_75143*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_item_catalog_embedding_layer_call_and_return_conditional_losses_7464720
.item_catalog_embedding/StatefulPartitionedCallА
*item_tag_embedding/StatefulPartitionedCallStatefulPartitionedCallitem_tagitem_tag_embedding_75146*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_item_tag_embedding_layer_call_and_return_conditional_losses_746682,
*item_tag_embedding/StatefulPartitionedCallТ
%embedding_concatenate/PartitionedCallPartitionedCall2user_id_embedding/StatefulPartitionedCall:output:06member_type_embedding/StatefulPartitionedCall:output:04user_type_embedding/StatefulPartitionedCall:output:02item_id_embedding/StatefulPartitionedCall:output:07item_catalog_embedding/StatefulPartitionedCall:output:03item_tag_embedding/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_embedding_concatenate_layer_call_and_return_conditional_losses_746912'
%embedding_concatenate/PartitionedCall
flatten_17/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_17_layer_call_and_return_conditional_losses_747102
flatten_17/PartitionedCallМ
"dnn_layer1/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dnn_layer1_75151dnn_layer1_75153*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer1_layer_call_and_return_conditional_losses_747412$
"dnn_layer1/StatefulPartitionedCall
dropout_24/PartitionedCallPartitionedCall+dnn_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_747742
dropout_24/PartitionedCallМ
"dnn_layer2/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0dnn_layer2_75157dnn_layer2_75159*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer2_layer_call_and_return_conditional_losses_748102$
"dnn_layer2/StatefulPartitionedCall
dropout_25/PartitionedCallPartitionedCall+dnn_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_748432
dropout_25/PartitionedCallМ
"dnn_layer3/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dnn_layer3_75163dnn_layer3_75165*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer3_layer_call_and_return_conditional_losses_748792$
"dnn_layer3/StatefulPartitionedCall
flatten_16/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_749012
flatten_16/PartitionedCall
dropout_26/PartitionedCallPartitionedCall+dnn_layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_26_layer_call_and_return_conditional_losses_749262
dropout_26/PartitionedCallЖ
!fm_linear/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0fm_linear_75170fm_linear_75172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_fm_linear_layer_call_and_return_conditional_losses_749612#
!fm_linear/StatefulPartitionedCallў
fm_cross/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fm_cross_layer_call_and_return_conditional_losses_749922
fm_cross/PartitionedCallЄ
fm_combine/PartitionedCallPartitionedCall*fm_linear/StatefulPartitionedCall:output:0!fm_cross/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fm_combine_layer_call_and_return_conditional_losses_750062
fm_combine/PartitionedCallЇ
"dnn_layer4/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0dnn_layer4_75177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer4_layer_call_and_return_conditional_losses_750282$
"dnn_layer4/StatefulPartitionedCall
add_8/PartitionedCallPartitionedCall#fm_combine/PartitionedCall:output:0+dnn_layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_8_layer_call_and_return_conditional_losses_750462
add_8/PartitionedCallы
sigmoid/PartitionedCallPartitionedCalladd_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_sigmoid_layer_call_and_return_conditional_losses_750602
sigmoid/PartitionedCallН
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer1_75151* 
_output_shapes
:
*
dtype025
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer1/kernel/Regularizer/SquareSquare;dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer1/kernel/Regularizer/Square
#dnn_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer1/kernel/Regularizer/ConstЦ
!dnn_layer1/kernel/Regularizer/SumSum(dnn_layer1/kernel/Regularizer/Square:y:0,dnn_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/Sum
#dnn_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer1/kernel/Regularizer/mul/xШ
!dnn_layer1/kernel/Regularizer/mulMul,dnn_layer1/kernel/Regularizer/mul/x:output:0*dnn_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/mulД
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer1_75153*
_output_shapes	
:*
dtype023
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer1/bias/Regularizer/SquareSquare9dnn_layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer1/bias/Regularizer/Square
!dnn_layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer1/bias/Regularizer/ConstО
dnn_layer1/bias/Regularizer/SumSum&dnn_layer1/bias/Regularizer/Square:y:0*dnn_layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/Sum
!dnn_layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer1/bias/Regularizer/mul/xР
dnn_layer1/bias/Regularizer/mulMul*dnn_layer1/bias/Regularizer/mul/x:output:0(dnn_layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/mulН
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer2_75157* 
_output_shapes
:
*
dtype025
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer2/kernel/Regularizer/SquareSquare;dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer2/kernel/Regularizer/Square
#dnn_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer2/kernel/Regularizer/ConstЦ
!dnn_layer2/kernel/Regularizer/SumSum(dnn_layer2/kernel/Regularizer/Square:y:0,dnn_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/Sum
#dnn_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer2/kernel/Regularizer/mul/xШ
!dnn_layer2/kernel/Regularizer/mulMul,dnn_layer2/kernel/Regularizer/mul/x:output:0*dnn_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/mulД
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer2_75159*
_output_shapes	
:*
dtype023
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer2/bias/Regularizer/SquareSquare9dnn_layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer2/bias/Regularizer/Square
!dnn_layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer2/bias/Regularizer/ConstО
dnn_layer2/bias/Regularizer/SumSum&dnn_layer2/bias/Regularizer/Square:y:0*dnn_layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/Sum
!dnn_layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer2/bias/Regularizer/mul/xР
dnn_layer2/bias/Regularizer/mulMul*dnn_layer2/bias/Regularizer/mul/x:output:0(dnn_layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/mulН
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer3_75163* 
_output_shapes
:
*
dtype025
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer3/kernel/Regularizer/SquareSquare;dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer3/kernel/Regularizer/Square
#dnn_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer3/kernel/Regularizer/ConstЦ
!dnn_layer3/kernel/Regularizer/SumSum(dnn_layer3/kernel/Regularizer/Square:y:0,dnn_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/Sum
#dnn_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer3/kernel/Regularizer/mul/xШ
!dnn_layer3/kernel/Regularizer/mulMul,dnn_layer3/kernel/Regularizer/mul/x:output:0*dnn_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/mulД
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer3_75165*
_output_shapes	
:*
dtype023
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer3/bias/Regularizer/SquareSquare9dnn_layer3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer3/bias/Regularizer/Square
!dnn_layer3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer3/bias/Regularizer/ConstО
dnn_layer3/bias/Regularizer/SumSum&dnn_layer3/bias/Regularizer/Square:y:0*dnn_layer3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/Sum
!dnn_layer3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer3/bias/Regularizer/mul/xР
dnn_layer3/bias/Regularizer/mulMul*dnn_layer3/bias/Regularizer/mul/x:output:0(dnn_layer3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/mulЙ
2fm_linear/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfm_linear_75170*
_output_shapes
:	*
dtype024
2fm_linear/kernel/Regularizer/Square/ReadVariableOpК
#fm_linear/kernel/Regularizer/SquareSquare:fm_linear/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#fm_linear/kernel/Regularizer/Square
"fm_linear/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"fm_linear/kernel/Regularizer/ConstТ
 fm_linear/kernel/Regularizer/SumSum'fm_linear/kernel/Regularizer/Square:y:0+fm_linear/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/Sum
"fm_linear/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2$
"fm_linear/kernel/Regularizer/mul/xФ
 fm_linear/kernel/Regularizer/mulMul+fm_linear/kernel/Regularizer/mul/x:output:0)fm_linear/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/mulА
0fm_linear/bias/Regularizer/Square/ReadVariableOpReadVariableOpfm_linear_75172*
_output_shapes
:*
dtype022
0fm_linear/bias/Regularizer/Square/ReadVariableOpЏ
!fm_linear/bias/Regularizer/SquareSquare8fm_linear/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!fm_linear/bias/Regularizer/Square
 fm_linear/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 fm_linear/bias/Regularizer/ConstК
fm_linear/bias/Regularizer/SumSum%fm_linear/bias/Regularizer/Square:y:0)fm_linear/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/Sum
 fm_linear/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2"
 fm_linear/bias/Regularizer/mul/xМ
fm_linear/bias/Regularizer/mulMul)fm_linear/bias/Regularizer/mul/x:output:0'fm_linear/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/mulМ
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer4_75177*
_output_shapes
:	*
dtype025
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpН
$dnn_layer4/kernel/Regularizer/SquareSquare;dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$dnn_layer4/kernel/Regularizer/Square
#dnn_layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer4/kernel/Regularizer/ConstЦ
!dnn_layer4/kernel/Regularizer/SumSum(dnn_layer4/kernel/Regularizer/Square:y:0,dnn_layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/Sum
#dnn_layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer4/kernel/Regularizer/mul/xШ
!dnn_layer4/kernel/Regularizer/mulMul,dnn_layer4/kernel/Regularizer/mul/x:output:0*dnn_layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/mul
IdentityIdentity sigmoid/PartitionedCall:output:0#^dnn_layer1/StatefulPartitionedCall2^dnn_layer1/bias/Regularizer/Square/ReadVariableOp4^dnn_layer1/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer2/StatefulPartitionedCall2^dnn_layer2/bias/Regularizer/Square/ReadVariableOp4^dnn_layer2/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer3/StatefulPartitionedCall2^dnn_layer3/bias/Regularizer/Square/ReadVariableOp4^dnn_layer3/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer4/StatefulPartitionedCall4^dnn_layer4/kernel/Regularizer/Square/ReadVariableOp"^fm_linear/StatefulPartitionedCall1^fm_linear/bias/Regularizer/Square/ReadVariableOp3^fm_linear/kernel/Regularizer/Square/ReadVariableOp/^item_catalog_embedding/StatefulPartitionedCall*^item_id_embedding/StatefulPartitionedCall+^item_tag_embedding/StatefulPartitionedCall.^member_type_embedding/StatefulPartitionedCall*^user_id_embedding/StatefulPartitionedCall,^user_type_embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::2H
"dnn_layer1/StatefulPartitionedCall"dnn_layer1/StatefulPartitionedCall2f
1dnn_layer1/bias/Regularizer/Square/ReadVariableOp1dnn_layer1/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer2/StatefulPartitionedCall"dnn_layer2/StatefulPartitionedCall2f
1dnn_layer2/bias/Regularizer/Square/ReadVariableOp1dnn_layer2/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer3/StatefulPartitionedCall"dnn_layer3/StatefulPartitionedCall2f
1dnn_layer3/bias/Regularizer/Square/ReadVariableOp1dnn_layer3/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer4/StatefulPartitionedCall"dnn_layer4/StatefulPartitionedCall2j
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp2F
!fm_linear/StatefulPartitionedCall!fm_linear/StatefulPartitionedCall2d
0fm_linear/bias/Regularizer/Square/ReadVariableOp0fm_linear/bias/Regularizer/Square/ReadVariableOp2h
2fm_linear/kernel/Regularizer/Square/ReadVariableOp2fm_linear/kernel/Regularizer/Square/ReadVariableOp2`
.item_catalog_embedding/StatefulPartitionedCall.item_catalog_embedding/StatefulPartitionedCall2V
)item_id_embedding/StatefulPartitionedCall)item_id_embedding/StatefulPartitionedCall2X
*item_tag_embedding/StatefulPartitionedCall*item_tag_embedding/StatefulPartitionedCall2^
-member_type_embedding/StatefulPartitionedCall-member_type_embedding/StatefulPartitionedCall2V
)user_id_embedding/StatefulPartitionedCall)user_id_embedding/StatefulPartitionedCall2Z
+user_type_embedding/StatefulPartitionedCall+user_type_embedding/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	user_id:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	user_type:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_namemember_type:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	item_id:UQ
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameitem_catalog:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
item_tag


 __inference__wrapped_model_74545
user_id
	user_type
member_type
item_id
item_catalog
item_tag5
1model_17_user_id_embedding_embedding_lookup_744619
5model_17_member_type_embedding_embedding_lookup_744667
3model_17_user_type_embedding_embedding_lookup_744715
1model_17_item_id_embedding_embedding_lookup_74476:
6model_17_item_catalog_embedding_embedding_lookup_744816
2model_17_item_tag_embedding_embedding_lookup_744866
2model_17_dnn_layer1_matmul_readvariableop_resource7
3model_17_dnn_layer1_biasadd_readvariableop_resource6
2model_17_dnn_layer2_matmul_readvariableop_resource7
3model_17_dnn_layer2_biasadd_readvariableop_resource6
2model_17_dnn_layer3_matmul_readvariableop_resource7
3model_17_dnn_layer3_biasadd_readvariableop_resource5
1model_17_fm_linear_matmul_readvariableop_resource6
2model_17_fm_linear_biasadd_readvariableop_resource6
2model_17_dnn_layer4_matmul_readvariableop_resource
identityЂ*model_17/dnn_layer1/BiasAdd/ReadVariableOpЂ)model_17/dnn_layer1/MatMul/ReadVariableOpЂ*model_17/dnn_layer2/BiasAdd/ReadVariableOpЂ)model_17/dnn_layer2/MatMul/ReadVariableOpЂ*model_17/dnn_layer3/BiasAdd/ReadVariableOpЂ)model_17/dnn_layer3/MatMul/ReadVariableOpЂ)model_17/dnn_layer4/MatMul/ReadVariableOpЂ)model_17/fm_linear/BiasAdd/ReadVariableOpЂ(model_17/fm_linear/MatMul/ReadVariableOpЂ0model_17/item_catalog_embedding/embedding_lookupЂ+model_17/item_id_embedding/embedding_lookupЂ,model_17/item_tag_embedding/embedding_lookupЂ/model_17/member_type_embedding/embedding_lookupЂ+model_17/user_id_embedding/embedding_lookupЂ-model_17/user_type_embedding/embedding_lookupц
+model_17/user_id_embedding/embedding_lookupResourceGather1model_17_user_id_embedding_embedding_lookup_74461user_id",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_17/user_id_embedding/embedding_lookup/74461*+
_output_shapes
:џџџџџџџџџ*
dtype02-
+model_17/user_id_embedding/embedding_lookupи
4model_17/user_id_embedding/embedding_lookup/IdentityIdentity4model_17/user_id_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_17/user_id_embedding/embedding_lookup/74461*+
_output_shapes
:џџџџџџџџџ26
4model_17/user_id_embedding/embedding_lookup/Identityё
6model_17/user_id_embedding/embedding_lookup/Identity_1Identity=model_17/user_id_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ28
6model_17/user_id_embedding/embedding_lookup/Identity_1њ
/model_17/member_type_embedding/embedding_lookupResourceGather5model_17_member_type_embedding_embedding_lookup_74466member_type",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*H
_class>
<:loc:@model_17/member_type_embedding/embedding_lookup/74466*+
_output_shapes
:џџџџџџџџџ*
dtype021
/model_17/member_type_embedding/embedding_lookupш
8model_17/member_type_embedding/embedding_lookup/IdentityIdentity8model_17/member_type_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@model_17/member_type_embedding/embedding_lookup/74466*+
_output_shapes
:џџџџџџџџџ2:
8model_17/member_type_embedding/embedding_lookup/Identity§
:model_17/member_type_embedding/embedding_lookup/Identity_1IdentityAmodel_17/member_type_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2<
:model_17/member_type_embedding/embedding_lookup/Identity_1№
-model_17/user_type_embedding/embedding_lookupResourceGather3model_17_user_type_embedding_embedding_lookup_74471	user_type",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*F
_class<
:8loc:@model_17/user_type_embedding/embedding_lookup/74471*+
_output_shapes
:џџџџџџџџџ*
dtype02/
-model_17/user_type_embedding/embedding_lookupр
6model_17/user_type_embedding/embedding_lookup/IdentityIdentity6model_17/user_type_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@model_17/user_type_embedding/embedding_lookup/74471*+
_output_shapes
:џџџџџџџџџ28
6model_17/user_type_embedding/embedding_lookup/Identityї
8model_17/user_type_embedding/embedding_lookup/Identity_1Identity?model_17/user_type_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2:
8model_17/user_type_embedding/embedding_lookup/Identity_1ц
+model_17/item_id_embedding/embedding_lookupResourceGather1model_17_item_id_embedding_embedding_lookup_74476item_id",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@model_17/item_id_embedding/embedding_lookup/74476*+
_output_shapes
:џџџџџџџџџ*
dtype02-
+model_17/item_id_embedding/embedding_lookupи
4model_17/item_id_embedding/embedding_lookup/IdentityIdentity4model_17/item_id_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@model_17/item_id_embedding/embedding_lookup/74476*+
_output_shapes
:џџџџџџџџџ26
4model_17/item_id_embedding/embedding_lookup/Identityё
6model_17/item_id_embedding/embedding_lookup/Identity_1Identity=model_17/item_id_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ28
6model_17/item_id_embedding/embedding_lookup/Identity_1џ
0model_17/item_catalog_embedding/embedding_lookupResourceGather6model_17_item_catalog_embedding_embedding_lookup_74481item_catalog",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*I
_class?
=;loc:@model_17/item_catalog_embedding/embedding_lookup/74481*+
_output_shapes
:џџџџџџџџџ*
dtype022
0model_17/item_catalog_embedding/embedding_lookupь
9model_17/item_catalog_embedding/embedding_lookup/IdentityIdentity9model_17/item_catalog_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*I
_class?
=;loc:@model_17/item_catalog_embedding/embedding_lookup/74481*+
_output_shapes
:џџџџџџџџџ2;
9model_17/item_catalog_embedding/embedding_lookup/Identity
;model_17/item_catalog_embedding/embedding_lookup/Identity_1IdentityBmodel_17/item_catalog_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2=
;model_17/item_catalog_embedding/embedding_lookup/Identity_1ы
,model_17/item_tag_embedding/embedding_lookupResourceGather2model_17_item_tag_embedding_embedding_lookup_74486item_tag",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*E
_class;
97loc:@model_17/item_tag_embedding/embedding_lookup/74486*+
_output_shapes
:џџџџџџџџџ*
dtype02.
,model_17/item_tag_embedding/embedding_lookupм
5model_17/item_tag_embedding/embedding_lookup/IdentityIdentity5model_17/item_tag_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@model_17/item_tag_embedding/embedding_lookup/74486*+
_output_shapes
:џџџџџџџџџ27
5model_17/item_tag_embedding/embedding_lookup/Identityє
7model_17/item_tag_embedding/embedding_lookup/Identity_1Identity>model_17/item_tag_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ29
7model_17/item_tag_embedding/embedding_lookup/Identity_1
*model_17/embedding_concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_17/embedding_concatenate/concat/axisр
%model_17/embedding_concatenate/concatConcatV2?model_17/user_id_embedding/embedding_lookup/Identity_1:output:0Cmodel_17/member_type_embedding/embedding_lookup/Identity_1:output:0Amodel_17/user_type_embedding/embedding_lookup/Identity_1:output:0?model_17/item_id_embedding/embedding_lookup/Identity_1:output:0Dmodel_17/item_catalog_embedding/embedding_lookup/Identity_1:output:0@model_17/item_tag_embedding/embedding_lookup/Identity_1:output:03model_17/embedding_concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ2'
%model_17/embedding_concatenate/concat
model_17/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
model_17/flatten_17/ConstЬ
model_17/flatten_17/ReshapeReshape.model_17/embedding_concatenate/concat:output:0"model_17/flatten_17/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/flatten_17/ReshapeЫ
)model_17/dnn_layer1/MatMul/ReadVariableOpReadVariableOp2model_17_dnn_layer1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)model_17/dnn_layer1/MatMul/ReadVariableOpЮ
model_17/dnn_layer1/MatMulMatMul$model_17/flatten_17/Reshape:output:01model_17/dnn_layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer1/MatMulЩ
*model_17/dnn_layer1/BiasAdd/ReadVariableOpReadVariableOp3model_17_dnn_layer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*model_17/dnn_layer1/BiasAdd/ReadVariableOpв
model_17/dnn_layer1/BiasAddBiasAdd$model_17/dnn_layer1/MatMul:product:02model_17/dnn_layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer1/BiasAdd
model_17/dnn_layer1/ReluRelu$model_17/dnn_layer1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer1/ReluЃ
model_17/dropout_24/IdentityIdentity&model_17/dnn_layer1/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dropout_24/IdentityЫ
)model_17/dnn_layer2/MatMul/ReadVariableOpReadVariableOp2model_17_dnn_layer2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)model_17/dnn_layer2/MatMul/ReadVariableOpЯ
model_17/dnn_layer2/MatMulMatMul%model_17/dropout_24/Identity:output:01model_17/dnn_layer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer2/MatMulЩ
*model_17/dnn_layer2/BiasAdd/ReadVariableOpReadVariableOp3model_17_dnn_layer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*model_17/dnn_layer2/BiasAdd/ReadVariableOpв
model_17/dnn_layer2/BiasAddBiasAdd$model_17/dnn_layer2/MatMul:product:02model_17/dnn_layer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer2/BiasAdd
model_17/dnn_layer2/ReluRelu$model_17/dnn_layer2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer2/ReluЃ
model_17/dropout_25/IdentityIdentity&model_17/dnn_layer2/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dropout_25/IdentityЫ
)model_17/dnn_layer3/MatMul/ReadVariableOpReadVariableOp2model_17_dnn_layer3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)model_17/dnn_layer3/MatMul/ReadVariableOpЯ
model_17/dnn_layer3/MatMulMatMul%model_17/dropout_25/Identity:output:01model_17/dnn_layer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer3/MatMulЩ
*model_17/dnn_layer3/BiasAdd/ReadVariableOpReadVariableOp3model_17_dnn_layer3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*model_17/dnn_layer3/BiasAdd/ReadVariableOpв
model_17/dnn_layer3/BiasAddBiasAdd$model_17/dnn_layer3/MatMul:product:02model_17/dnn_layer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer3/BiasAdd
model_17/dnn_layer3/ReluRelu$model_17/dnn_layer3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer3/Relu
model_17/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
model_17/flatten_16/ConstЬ
model_17/flatten_16/ReshapeReshape.model_17/embedding_concatenate/concat:output:0"model_17/flatten_16/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/flatten_16/ReshapeЃ
model_17/dropout_26/IdentityIdentity&model_17/dnn_layer3/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_17/dropout_26/IdentityЧ
(model_17/fm_linear/MatMul/ReadVariableOpReadVariableOp1model_17_fm_linear_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(model_17/fm_linear/MatMul/ReadVariableOpЪ
model_17/fm_linear/MatMulMatMul$model_17/flatten_16/Reshape:output:00model_17/fm_linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_17/fm_linear/MatMulХ
)model_17/fm_linear/BiasAdd/ReadVariableOpReadVariableOp2model_17_fm_linear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_17/fm_linear/BiasAdd/ReadVariableOpЭ
model_17/fm_linear/BiasAddBiasAdd#model_17/fm_linear/MatMul:product:01model_17/fm_linear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_17/fm_linear/BiasAdd
'model_17/fm_cross/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_17/fm_cross/Sum/reduction_indicesо
model_17/fm_cross/SumSum.model_17/embedding_concatenate/concat:output:00model_17/fm_cross/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
model_17/fm_cross/Sum
model_17/fm_cross/SquareSquaremodel_17/fm_cross/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
model_17/fm_cross/SquareЈ
model_17/fm_cross/Square_1Square.model_17/embedding_concatenate/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
model_17/fm_cross/Square_1
)model_17/fm_cross/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_17/fm_cross/Sum_1/reduction_indicesд
model_17/fm_cross/Sum_1Summodel_17/fm_cross/Square_1:y:02model_17/fm_cross/Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
model_17/fm_cross/Sum_1Ћ
model_17/fm_cross/subSubmodel_17/fm_cross/Square:y:0 model_17/fm_cross/Sum_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
model_17/fm_cross/sub
)model_17/fm_cross/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_17/fm_cross/Sum_2/reduction_indicesК
model_17/fm_cross/Sum_2Summodel_17/fm_cross/sub:z:02model_17/fm_cross/Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_17/fm_cross/Sum_2w
model_17/fm_cross/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_17/fm_cross/mul/xЋ
model_17/fm_cross/mulMul model_17/fm_cross/mul/x:output:0 model_17/fm_cross/Sum_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_17/fm_cross/mul­
model_17/fm_combine/addAddV2#model_17/fm_linear/BiasAdd:output:0model_17/fm_cross/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_17/fm_combine/addЪ
)model_17/dnn_layer4/MatMul/ReadVariableOpReadVariableOp2model_17_dnn_layer4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)model_17/dnn_layer4/MatMul/ReadVariableOpЮ
model_17/dnn_layer4/MatMulMatMul%model_17/dropout_26/Identity:output:01model_17/dnn_layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_17/dnn_layer4/MatMulІ
model_17/add_8/addAddV2model_17/fm_combine/add:z:0$model_17/dnn_layer4/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_17/add_8/add
model_17/sigmoid/SigmoidSigmoidmodel_17/add_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_17/sigmoid/Sigmoid
IdentityIdentitymodel_17/sigmoid/Sigmoid:y:0+^model_17/dnn_layer1/BiasAdd/ReadVariableOp*^model_17/dnn_layer1/MatMul/ReadVariableOp+^model_17/dnn_layer2/BiasAdd/ReadVariableOp*^model_17/dnn_layer2/MatMul/ReadVariableOp+^model_17/dnn_layer3/BiasAdd/ReadVariableOp*^model_17/dnn_layer3/MatMul/ReadVariableOp*^model_17/dnn_layer4/MatMul/ReadVariableOp*^model_17/fm_linear/BiasAdd/ReadVariableOp)^model_17/fm_linear/MatMul/ReadVariableOp1^model_17/item_catalog_embedding/embedding_lookup,^model_17/item_id_embedding/embedding_lookup-^model_17/item_tag_embedding/embedding_lookup0^model_17/member_type_embedding/embedding_lookup,^model_17/user_id_embedding/embedding_lookup.^model_17/user_type_embedding/embedding_lookup*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::2X
*model_17/dnn_layer1/BiasAdd/ReadVariableOp*model_17/dnn_layer1/BiasAdd/ReadVariableOp2V
)model_17/dnn_layer1/MatMul/ReadVariableOp)model_17/dnn_layer1/MatMul/ReadVariableOp2X
*model_17/dnn_layer2/BiasAdd/ReadVariableOp*model_17/dnn_layer2/BiasAdd/ReadVariableOp2V
)model_17/dnn_layer2/MatMul/ReadVariableOp)model_17/dnn_layer2/MatMul/ReadVariableOp2X
*model_17/dnn_layer3/BiasAdd/ReadVariableOp*model_17/dnn_layer3/BiasAdd/ReadVariableOp2V
)model_17/dnn_layer3/MatMul/ReadVariableOp)model_17/dnn_layer3/MatMul/ReadVariableOp2V
)model_17/dnn_layer4/MatMul/ReadVariableOp)model_17/dnn_layer4/MatMul/ReadVariableOp2V
)model_17/fm_linear/BiasAdd/ReadVariableOp)model_17/fm_linear/BiasAdd/ReadVariableOp2T
(model_17/fm_linear/MatMul/ReadVariableOp(model_17/fm_linear/MatMul/ReadVariableOp2d
0model_17/item_catalog_embedding/embedding_lookup0model_17/item_catalog_embedding/embedding_lookup2Z
+model_17/item_id_embedding/embedding_lookup+model_17/item_id_embedding/embedding_lookup2\
,model_17/item_tag_embedding/embedding_lookup,model_17/item_tag_embedding/embedding_lookup2b
/model_17/member_type_embedding/embedding_lookup/model_17/member_type_embedding/embedding_lookup2Z
+model_17/user_id_embedding/embedding_lookup+model_17/user_id_embedding/embedding_lookup2^
-model_17/user_type_embedding/embedding_lookup-model_17/user_type_embedding/embedding_lookup:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	user_id:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	user_type:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_namemember_type:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	item_id:UQ
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameitem_catalog:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
item_tag
п
І
__inference_loss_fn_2_76551@
<dnn_layer2_kernel_regularizer_square_readvariableop_resource
identityЂ3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpщ
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<dnn_layer2_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer2/kernel/Regularizer/SquareSquare;dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer2/kernel/Regularizer/Square
#dnn_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer2/kernel/Regularizer/ConstЦ
!dnn_layer2/kernel/Regularizer/SumSum(dnn_layer2/kernel/Regularizer/Square:y:0,dnn_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/Sum
#dnn_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer2/kernel/Regularizer/mul/xШ
!dnn_layer2/kernel/Regularizer/mulMul,dnn_layer2/kernel/Regularizer/mul/x:output:0*dnn_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/mul
IdentityIdentity%dnn_layer2/kernel/Regularizer/mul:z:04^dnn_layer2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2j
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp
Е
a
E__inference_flatten_17_layer_call_and_return_conditional_losses_76166

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
п
І
__inference_loss_fn_4_76573@
<dnn_layer3_kernel_regularizer_square_readvariableop_resource
identityЂ3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpщ
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<dnn_layer3_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer3/kernel/Regularizer/SquareSquare;dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer3/kernel/Regularizer/Square
#dnn_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer3/kernel/Regularizer/ConstЦ
!dnn_layer3/kernel/Regularizer/SumSum(dnn_layer3/kernel/Regularizer/Square:y:0,dnn_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/Sum
#dnn_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer3/kernel/Regularizer/mul/xШ
!dnn_layer3/kernel/Regularizer/mulMul,dnn_layer3/kernel/Regularizer/mul/x:output:0*dnn_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/mul
IdentityIdentity%dnn_layer3/kernel/Regularizer/mul:z:04^dnn_layer3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2j
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp
Ь
c
E__inference_dropout_26_layer_call_and_return_conditional_losses_76448

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
E__inference_dropout_25_layer_call_and_return_conditional_losses_74838

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
Х
C__inference_model_17_layer_call_and_return_conditional_losses_75123
user_id
	user_type
member_type
item_id
item_catalog
item_tag
user_id_embedding_74572
member_type_embedding_74593
user_type_embedding_74614
item_id_embedding_74635 
item_catalog_embedding_74656
item_tag_embedding_74677
dnn_layer1_74752
dnn_layer1_74754
dnn_layer2_74821
dnn_layer2_74823
dnn_layer3_74890
dnn_layer3_74892
fm_linear_74972
fm_linear_74974
dnn_layer4_75037
identityЂ"dnn_layer1/StatefulPartitionedCallЂ1dnn_layer1/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer2/StatefulPartitionedCallЂ1dnn_layer2/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer3/StatefulPartitionedCallЂ1dnn_layer3/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer4/StatefulPartitionedCallЂ3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpЂ"dropout_24/StatefulPartitionedCallЂ"dropout_25/StatefulPartitionedCallЂ"dropout_26/StatefulPartitionedCallЂ!fm_linear/StatefulPartitionedCallЂ0fm_linear/bias/Regularizer/Square/ReadVariableOpЂ2fm_linear/kernel/Regularizer/Square/ReadVariableOpЂ.item_catalog_embedding/StatefulPartitionedCallЂ)item_id_embedding/StatefulPartitionedCallЂ*item_tag_embedding/StatefulPartitionedCallЂ-member_type_embedding/StatefulPartitionedCallЂ)user_id_embedding/StatefulPartitionedCallЂ+user_type_embedding/StatefulPartitionedCallЋ
)user_id_embedding/StatefulPartitionedCallStatefulPartitionedCalluser_iduser_id_embedding_74572*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_user_id_embedding_layer_call_and_return_conditional_losses_745632+
)user_id_embedding/StatefulPartitionedCallП
-member_type_embedding/StatefulPartitionedCallStatefulPartitionedCallmember_typemember_type_embedding_74593*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_member_type_embedding_layer_call_and_return_conditional_losses_745842/
-member_type_embedding/StatefulPartitionedCallЕ
+user_type_embedding/StatefulPartitionedCallStatefulPartitionedCall	user_typeuser_type_embedding_74614*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_user_type_embedding_layer_call_and_return_conditional_losses_746052-
+user_type_embedding/StatefulPartitionedCallЋ
)item_id_embedding/StatefulPartitionedCallStatefulPartitionedCallitem_iditem_id_embedding_74635*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_item_id_embedding_layer_call_and_return_conditional_losses_746262+
)item_id_embedding/StatefulPartitionedCallФ
.item_catalog_embedding/StatefulPartitionedCallStatefulPartitionedCallitem_catalogitem_catalog_embedding_74656*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_item_catalog_embedding_layer_call_and_return_conditional_losses_7464720
.item_catalog_embedding/StatefulPartitionedCallА
*item_tag_embedding/StatefulPartitionedCallStatefulPartitionedCallitem_tagitem_tag_embedding_74677*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_item_tag_embedding_layer_call_and_return_conditional_losses_746682,
*item_tag_embedding/StatefulPartitionedCallТ
%embedding_concatenate/PartitionedCallPartitionedCall2user_id_embedding/StatefulPartitionedCall:output:06member_type_embedding/StatefulPartitionedCall:output:04user_type_embedding/StatefulPartitionedCall:output:02item_id_embedding/StatefulPartitionedCall:output:07item_catalog_embedding/StatefulPartitionedCall:output:03item_tag_embedding/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_embedding_concatenate_layer_call_and_return_conditional_losses_746912'
%embedding_concatenate/PartitionedCall
flatten_17/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_17_layer_call_and_return_conditional_losses_747102
flatten_17/PartitionedCallМ
"dnn_layer1/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dnn_layer1_74752dnn_layer1_74754*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer1_layer_call_and_return_conditional_losses_747412$
"dnn_layer1/StatefulPartitionedCall
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall+dnn_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_747692$
"dropout_24/StatefulPartitionedCallФ
"dnn_layer2/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0dnn_layer2_74821dnn_layer2_74823*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer2_layer_call_and_return_conditional_losses_748102$
"dnn_layer2/StatefulPartitionedCallП
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall+dnn_layer2/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_748382$
"dropout_25/StatefulPartitionedCallФ
"dnn_layer3/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dnn_layer3_74890dnn_layer3_74892*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer3_layer_call_and_return_conditional_losses_748792$
"dnn_layer3/StatefulPartitionedCall
flatten_16/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_749012
flatten_16/PartitionedCallП
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall+dnn_layer3/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_26_layer_call_and_return_conditional_losses_749212$
"dropout_26/StatefulPartitionedCallЖ
!fm_linear/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0fm_linear_74972fm_linear_74974*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_fm_linear_layer_call_and_return_conditional_losses_749612#
!fm_linear/StatefulPartitionedCallў
fm_cross/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fm_cross_layer_call_and_return_conditional_losses_749922
fm_cross/PartitionedCallЄ
fm_combine/PartitionedCallPartitionedCall*fm_linear/StatefulPartitionedCall:output:0!fm_cross/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fm_combine_layer_call_and_return_conditional_losses_750062
fm_combine/PartitionedCallЏ
"dnn_layer4/StatefulPartitionedCallStatefulPartitionedCall+dropout_26/StatefulPartitionedCall:output:0dnn_layer4_75037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer4_layer_call_and_return_conditional_losses_750282$
"dnn_layer4/StatefulPartitionedCall
add_8/PartitionedCallPartitionedCall#fm_combine/PartitionedCall:output:0+dnn_layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_8_layer_call_and_return_conditional_losses_750462
add_8/PartitionedCallы
sigmoid/PartitionedCallPartitionedCalladd_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_sigmoid_layer_call_and_return_conditional_losses_750602
sigmoid/PartitionedCallН
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer1_74752* 
_output_shapes
:
*
dtype025
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer1/kernel/Regularizer/SquareSquare;dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer1/kernel/Regularizer/Square
#dnn_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer1/kernel/Regularizer/ConstЦ
!dnn_layer1/kernel/Regularizer/SumSum(dnn_layer1/kernel/Regularizer/Square:y:0,dnn_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/Sum
#dnn_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer1/kernel/Regularizer/mul/xШ
!dnn_layer1/kernel/Regularizer/mulMul,dnn_layer1/kernel/Regularizer/mul/x:output:0*dnn_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/mulД
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer1_74754*
_output_shapes	
:*
dtype023
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer1/bias/Regularizer/SquareSquare9dnn_layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer1/bias/Regularizer/Square
!dnn_layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer1/bias/Regularizer/ConstО
dnn_layer1/bias/Regularizer/SumSum&dnn_layer1/bias/Regularizer/Square:y:0*dnn_layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/Sum
!dnn_layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer1/bias/Regularizer/mul/xР
dnn_layer1/bias/Regularizer/mulMul*dnn_layer1/bias/Regularizer/mul/x:output:0(dnn_layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/mulН
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer2_74821* 
_output_shapes
:
*
dtype025
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer2/kernel/Regularizer/SquareSquare;dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer2/kernel/Regularizer/Square
#dnn_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer2/kernel/Regularizer/ConstЦ
!dnn_layer2/kernel/Regularizer/SumSum(dnn_layer2/kernel/Regularizer/Square:y:0,dnn_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/Sum
#dnn_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer2/kernel/Regularizer/mul/xШ
!dnn_layer2/kernel/Regularizer/mulMul,dnn_layer2/kernel/Regularizer/mul/x:output:0*dnn_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/mulД
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer2_74823*
_output_shapes	
:*
dtype023
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer2/bias/Regularizer/SquareSquare9dnn_layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer2/bias/Regularizer/Square
!dnn_layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer2/bias/Regularizer/ConstО
dnn_layer2/bias/Regularizer/SumSum&dnn_layer2/bias/Regularizer/Square:y:0*dnn_layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/Sum
!dnn_layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer2/bias/Regularizer/mul/xР
dnn_layer2/bias/Regularizer/mulMul*dnn_layer2/bias/Regularizer/mul/x:output:0(dnn_layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/mulН
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer3_74890* 
_output_shapes
:
*
dtype025
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer3/kernel/Regularizer/SquareSquare;dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer3/kernel/Regularizer/Square
#dnn_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer3/kernel/Regularizer/ConstЦ
!dnn_layer3/kernel/Regularizer/SumSum(dnn_layer3/kernel/Regularizer/Square:y:0,dnn_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/Sum
#dnn_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer3/kernel/Regularizer/mul/xШ
!dnn_layer3/kernel/Regularizer/mulMul,dnn_layer3/kernel/Regularizer/mul/x:output:0*dnn_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/mulД
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer3_74892*
_output_shapes	
:*
dtype023
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer3/bias/Regularizer/SquareSquare9dnn_layer3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer3/bias/Regularizer/Square
!dnn_layer3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer3/bias/Regularizer/ConstО
dnn_layer3/bias/Regularizer/SumSum&dnn_layer3/bias/Regularizer/Square:y:0*dnn_layer3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/Sum
!dnn_layer3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer3/bias/Regularizer/mul/xР
dnn_layer3/bias/Regularizer/mulMul*dnn_layer3/bias/Regularizer/mul/x:output:0(dnn_layer3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/mulЙ
2fm_linear/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfm_linear_74972*
_output_shapes
:	*
dtype024
2fm_linear/kernel/Regularizer/Square/ReadVariableOpК
#fm_linear/kernel/Regularizer/SquareSquare:fm_linear/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#fm_linear/kernel/Regularizer/Square
"fm_linear/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"fm_linear/kernel/Regularizer/ConstТ
 fm_linear/kernel/Regularizer/SumSum'fm_linear/kernel/Regularizer/Square:y:0+fm_linear/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/Sum
"fm_linear/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2$
"fm_linear/kernel/Regularizer/mul/xФ
 fm_linear/kernel/Regularizer/mulMul+fm_linear/kernel/Regularizer/mul/x:output:0)fm_linear/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/mulА
0fm_linear/bias/Regularizer/Square/ReadVariableOpReadVariableOpfm_linear_74974*
_output_shapes
:*
dtype022
0fm_linear/bias/Regularizer/Square/ReadVariableOpЏ
!fm_linear/bias/Regularizer/SquareSquare8fm_linear/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!fm_linear/bias/Regularizer/Square
 fm_linear/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 fm_linear/bias/Regularizer/ConstК
fm_linear/bias/Regularizer/SumSum%fm_linear/bias/Regularizer/Square:y:0)fm_linear/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/Sum
 fm_linear/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2"
 fm_linear/bias/Regularizer/mul/xМ
fm_linear/bias/Regularizer/mulMul)fm_linear/bias/Regularizer/mul/x:output:0'fm_linear/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/mulМ
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer4_75037*
_output_shapes
:	*
dtype025
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpН
$dnn_layer4/kernel/Regularizer/SquareSquare;dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$dnn_layer4/kernel/Regularizer/Square
#dnn_layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer4/kernel/Regularizer/ConstЦ
!dnn_layer4/kernel/Regularizer/SumSum(dnn_layer4/kernel/Regularizer/Square:y:0,dnn_layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/Sum
#dnn_layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer4/kernel/Regularizer/mul/xШ
!dnn_layer4/kernel/Regularizer/mulMul,dnn_layer4/kernel/Regularizer/mul/x:output:0*dnn_layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/mul	
IdentityIdentity sigmoid/PartitionedCall:output:0#^dnn_layer1/StatefulPartitionedCall2^dnn_layer1/bias/Regularizer/Square/ReadVariableOp4^dnn_layer1/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer2/StatefulPartitionedCall2^dnn_layer2/bias/Regularizer/Square/ReadVariableOp4^dnn_layer2/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer3/StatefulPartitionedCall2^dnn_layer3/bias/Regularizer/Square/ReadVariableOp4^dnn_layer3/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer4/StatefulPartitionedCall4^dnn_layer4/kernel/Regularizer/Square/ReadVariableOp#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall"^fm_linear/StatefulPartitionedCall1^fm_linear/bias/Regularizer/Square/ReadVariableOp3^fm_linear/kernel/Regularizer/Square/ReadVariableOp/^item_catalog_embedding/StatefulPartitionedCall*^item_id_embedding/StatefulPartitionedCall+^item_tag_embedding/StatefulPartitionedCall.^member_type_embedding/StatefulPartitionedCall*^user_id_embedding/StatefulPartitionedCall,^user_type_embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::2H
"dnn_layer1/StatefulPartitionedCall"dnn_layer1/StatefulPartitionedCall2f
1dnn_layer1/bias/Regularizer/Square/ReadVariableOp1dnn_layer1/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer2/StatefulPartitionedCall"dnn_layer2/StatefulPartitionedCall2f
1dnn_layer2/bias/Regularizer/Square/ReadVariableOp1dnn_layer2/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer3/StatefulPartitionedCall"dnn_layer3/StatefulPartitionedCall2f
1dnn_layer3/bias/Regularizer/Square/ReadVariableOp1dnn_layer3/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer4/StatefulPartitionedCall"dnn_layer4/StatefulPartitionedCall2j
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall2F
!fm_linear/StatefulPartitionedCall!fm_linear/StatefulPartitionedCall2d
0fm_linear/bias/Regularizer/Square/ReadVariableOp0fm_linear/bias/Regularizer/Square/ReadVariableOp2h
2fm_linear/kernel/Regularizer/Square/ReadVariableOp2fm_linear/kernel/Regularizer/Square/ReadVariableOp2`
.item_catalog_embedding/StatefulPartitionedCall.item_catalog_embedding/StatefulPartitionedCall2V
)item_id_embedding/StatefulPartitionedCall)item_id_embedding/StatefulPartitionedCall2X
*item_tag_embedding/StatefulPartitionedCall*item_tag_embedding/StatefulPartitionedCall2^
-member_type_embedding/StatefulPartitionedCall-member_type_embedding/StatefulPartitionedCall2V
)user_id_embedding/StatefulPartitionedCall)user_id_embedding/StatefulPartitionedCall2Z
+user_type_embedding/StatefulPartitionedCall+user_type_embedding/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	user_id:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	user_type:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_namemember_type:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	item_id:UQ
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameitem_catalog:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
item_tag

F
*__inference_dropout_25_layer_call_fn_76313

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_748432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
^
B__inference_sigmoid_layer_call_and_return_conditional_losses_76513

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

_
C__inference_fm_cross_layer_call_and_return_conditional_losses_74992

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSuminputsSum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
Sum^
SquareSquareSum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Square\
Square_1Squareinputs*
T0*+
_output_shapes
:џџџџџџџџџ2

Square_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indices
Sum_1SumSquare_1:y:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
Sum_1c
subSub
Square:y:0Sum_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
subt
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_2/reduction_indicesr
Sum_2Sumsub:z:0 Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Sum_2S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xc
mulMulmul/x:output:0Sum_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
E__inference_dropout_25_layer_call_and_return_conditional_losses_76298

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х
Є
__inference_loss_fn_6_76595?
;fm_linear_kernel_regularizer_square_readvariableop_resource
identityЂ2fm_linear/kernel/Regularizer/Square/ReadVariableOpх
2fm_linear/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;fm_linear_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype024
2fm_linear/kernel/Regularizer/Square/ReadVariableOpК
#fm_linear/kernel/Regularizer/SquareSquare:fm_linear/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#fm_linear/kernel/Regularizer/Square
"fm_linear/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"fm_linear/kernel/Regularizer/ConstТ
 fm_linear/kernel/Regularizer/SumSum'fm_linear/kernel/Regularizer/Square:y:0+fm_linear/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/Sum
"fm_linear/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2$
"fm_linear/kernel/Regularizer/mul/xФ
 fm_linear/kernel/Regularizer/mulMul+fm_linear/kernel/Regularizer/mul/x:output:0)fm_linear/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/mul
IdentityIdentity$fm_linear/kernel/Regularizer/mul:z:03^fm_linear/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2h
2fm_linear/kernel/Regularizer/Square/ReadVariableOp2fm_linear/kernel/Regularizer/Square/ReadVariableOp
Ь
c
E__inference_dropout_26_layer_call_and_return_conditional_losses_74926

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
a
E__inference_flatten_16_layer_call_and_return_conditional_losses_76319

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

_
C__inference_fm_cross_layer_call_and_return_conditional_losses_76426

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices
SumSuminputsSum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
Sum^
SquareSquareSum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
Square\
Square_1Squareinputs*
T0*+
_output_shapes
:џџџџџџџџџ2

Square_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indices
Sum_1SumSquare_1:y:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
Sum_1c
subSub
Square:y:0Sum_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
subt
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_2/reduction_indicesr
Sum_2Sumsub:z:0 Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Sum_2S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xc
mulMulmul/x:output:0Sum_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul[
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

D
(__inference_fm_cross_layer_call_fn_76431

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fm_cross_layer_call_and_return_conditional_losses_749922
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
юї
і
C__inference_model_17_layer_call_and_return_conditional_losses_75817
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5,
(user_id_embedding_embedding_lookup_756580
,member_type_embedding_embedding_lookup_75663.
*user_type_embedding_embedding_lookup_75668,
(item_id_embedding_embedding_lookup_756731
-item_catalog_embedding_embedding_lookup_75678-
)item_tag_embedding_embedding_lookup_75683-
)dnn_layer1_matmul_readvariableop_resource.
*dnn_layer1_biasadd_readvariableop_resource-
)dnn_layer2_matmul_readvariableop_resource.
*dnn_layer2_biasadd_readvariableop_resource-
)dnn_layer3_matmul_readvariableop_resource.
*dnn_layer3_biasadd_readvariableop_resource,
(fm_linear_matmul_readvariableop_resource-
)fm_linear_biasadd_readvariableop_resource-
)dnn_layer4_matmul_readvariableop_resource
identityЂ!dnn_layer1/BiasAdd/ReadVariableOpЂ dnn_layer1/MatMul/ReadVariableOpЂ1dnn_layer1/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpЂ!dnn_layer2/BiasAdd/ReadVariableOpЂ dnn_layer2/MatMul/ReadVariableOpЂ1dnn_layer2/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpЂ!dnn_layer3/BiasAdd/ReadVariableOpЂ dnn_layer3/MatMul/ReadVariableOpЂ1dnn_layer3/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpЂ dnn_layer4/MatMul/ReadVariableOpЂ3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpЂ fm_linear/BiasAdd/ReadVariableOpЂfm_linear/MatMul/ReadVariableOpЂ0fm_linear/bias/Regularizer/Square/ReadVariableOpЂ2fm_linear/kernel/Regularizer/Square/ReadVariableOpЂ'item_catalog_embedding/embedding_lookupЂ"item_id_embedding/embedding_lookupЂ#item_tag_embedding/embedding_lookupЂ&member_type_embedding/embedding_lookupЂ"user_id_embedding/embedding_lookupЂ$user_type_embedding/embedding_lookupУ
"user_id_embedding/embedding_lookupResourceGather(user_id_embedding_embedding_lookup_75658inputs_0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@user_id_embedding/embedding_lookup/75658*+
_output_shapes
:џџџџџџџџџ*
dtype02$
"user_id_embedding/embedding_lookupД
+user_id_embedding/embedding_lookup/IdentityIdentity+user_id_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@user_id_embedding/embedding_lookup/75658*+
_output_shapes
:џџџџџџџџџ2-
+user_id_embedding/embedding_lookup/Identityж
-user_id_embedding/embedding_lookup/Identity_1Identity4user_id_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2/
-user_id_embedding/embedding_lookup/Identity_1г
&member_type_embedding/embedding_lookupResourceGather,member_type_embedding_embedding_lookup_75663inputs_2",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*?
_class5
31loc:@member_type_embedding/embedding_lookup/75663*+
_output_shapes
:џџџџџџџџџ*
dtype02(
&member_type_embedding/embedding_lookupФ
/member_type_embedding/embedding_lookup/IdentityIdentity/member_type_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@member_type_embedding/embedding_lookup/75663*+
_output_shapes
:џџџџџџџџџ21
/member_type_embedding/embedding_lookup/Identityт
1member_type_embedding/embedding_lookup/Identity_1Identity8member_type_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ23
1member_type_embedding/embedding_lookup/Identity_1Ы
$user_type_embedding/embedding_lookupResourceGather*user_type_embedding_embedding_lookup_75668inputs_1",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*=
_class3
1/loc:@user_type_embedding/embedding_lookup/75668*+
_output_shapes
:џџџџџџџџџ*
dtype02&
$user_type_embedding/embedding_lookupМ
-user_type_embedding/embedding_lookup/IdentityIdentity-user_type_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@user_type_embedding/embedding_lookup/75668*+
_output_shapes
:џџџџџџџџџ2/
-user_type_embedding/embedding_lookup/Identityм
/user_type_embedding/embedding_lookup/Identity_1Identity6user_type_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ21
/user_type_embedding/embedding_lookup/Identity_1У
"item_id_embedding/embedding_lookupResourceGather(item_id_embedding_embedding_lookup_75673inputs_3",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@item_id_embedding/embedding_lookup/75673*+
_output_shapes
:џџџџџџџџџ*
dtype02$
"item_id_embedding/embedding_lookupД
+item_id_embedding/embedding_lookup/IdentityIdentity+item_id_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@item_id_embedding/embedding_lookup/75673*+
_output_shapes
:џџџџџџџџџ2-
+item_id_embedding/embedding_lookup/Identityж
-item_id_embedding/embedding_lookup/Identity_1Identity4item_id_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2/
-item_id_embedding/embedding_lookup/Identity_1з
'item_catalog_embedding/embedding_lookupResourceGather-item_catalog_embedding_embedding_lookup_75678inputs_4",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*@
_class6
42loc:@item_catalog_embedding/embedding_lookup/75678*+
_output_shapes
:џџџџџџџџџ*
dtype02)
'item_catalog_embedding/embedding_lookupШ
0item_catalog_embedding/embedding_lookup/IdentityIdentity0item_catalog_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@item_catalog_embedding/embedding_lookup/75678*+
_output_shapes
:џџџџџџџџџ22
0item_catalog_embedding/embedding_lookup/Identityх
2item_catalog_embedding/embedding_lookup/Identity_1Identity9item_catalog_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ24
2item_catalog_embedding/embedding_lookup/Identity_1Ч
#item_tag_embedding/embedding_lookupResourceGather)item_tag_embedding_embedding_lookup_75683inputs_5",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@item_tag_embedding/embedding_lookup/75683*+
_output_shapes
:џџџџџџџџџ*
dtype02%
#item_tag_embedding/embedding_lookupИ
,item_tag_embedding/embedding_lookup/IdentityIdentity,item_tag_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@item_tag_embedding/embedding_lookup/75683*+
_output_shapes
:џџџџџџџџџ2.
,item_tag_embedding/embedding_lookup/Identityй
.item_tag_embedding/embedding_lookup/Identity_1Identity5item_tag_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ20
.item_tag_embedding/embedding_lookup/Identity_1
!embedding_concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!embedding_concatenate/concat/axis
embedding_concatenate/concatConcatV26user_id_embedding/embedding_lookup/Identity_1:output:0:member_type_embedding/embedding_lookup/Identity_1:output:08user_type_embedding/embedding_lookup/Identity_1:output:06item_id_embedding/embedding_lookup/Identity_1:output:0;item_catalog_embedding/embedding_lookup/Identity_1:output:07item_tag_embedding/embedding_lookup/Identity_1:output:0*embedding_concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_concatenate/concatu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_17/ConstЈ
flatten_17/ReshapeReshape%embedding_concatenate/concat:output:0flatten_17/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_17/ReshapeА
 dnn_layer1/MatMul/ReadVariableOpReadVariableOp)dnn_layer1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dnn_layer1/MatMul/ReadVariableOpЊ
dnn_layer1/MatMulMatMulflatten_17/Reshape:output:0(dnn_layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer1/MatMulЎ
!dnn_layer1/BiasAdd/ReadVariableOpReadVariableOp*dnn_layer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dnn_layer1/BiasAdd/ReadVariableOpЎ
dnn_layer1/BiasAddBiasAdddnn_layer1/MatMul:product:0)dnn_layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer1/BiasAddz
dnn_layer1/ReluReludnn_layer1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer1/Reluy
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout_24/dropout/ConstЌ
dropout_24/dropout/MulMuldnn_layer1/Relu:activations:0!dropout_24/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_24/dropout/Mul
dropout_24/dropout/ShapeShapednn_layer1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_24/dropout/Shapeж
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype021
/dropout_24/dropout/random_uniform/RandomUniform
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2#
!dropout_24/dropout/GreaterEqual/yы
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
dropout_24/dropout/GreaterEqualЁ
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout_24/dropout/CastЇ
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_24/dropout/Mul_1А
 dnn_layer2/MatMul/ReadVariableOpReadVariableOp)dnn_layer2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dnn_layer2/MatMul/ReadVariableOpЋ
dnn_layer2/MatMulMatMuldropout_24/dropout/Mul_1:z:0(dnn_layer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer2/MatMulЎ
!dnn_layer2/BiasAdd/ReadVariableOpReadVariableOp*dnn_layer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dnn_layer2/BiasAdd/ReadVariableOpЎ
dnn_layer2/BiasAddBiasAdddnn_layer2/MatMul:product:0)dnn_layer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer2/BiasAddz
dnn_layer2/ReluReludnn_layer2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer2/Reluy
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout_25/dropout/ConstЌ
dropout_25/dropout/MulMuldnn_layer2/Relu:activations:0!dropout_25/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_25/dropout/Mul
dropout_25/dropout/ShapeShapednn_layer2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_25/dropout/Shapeж
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype021
/dropout_25/dropout/random_uniform/RandomUniform
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2#
!dropout_25/dropout/GreaterEqual/yы
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
dropout_25/dropout/GreaterEqualЁ
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout_25/dropout/CastЇ
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_25/dropout/Mul_1А
 dnn_layer3/MatMul/ReadVariableOpReadVariableOp)dnn_layer3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 dnn_layer3/MatMul/ReadVariableOpЋ
dnn_layer3/MatMulMatMuldropout_25/dropout/Mul_1:z:0(dnn_layer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer3/MatMulЎ
!dnn_layer3/BiasAdd/ReadVariableOpReadVariableOp*dnn_layer3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dnn_layer3/BiasAdd/ReadVariableOpЎ
dnn_layer3/BiasAddBiasAdddnn_layer3/MatMul:product:0)dnn_layer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer3/BiasAddz
dnn_layer3/ReluReludnn_layer3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dnn_layer3/Reluu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_16/ConstЈ
flatten_16/ReshapeReshape%embedding_concatenate/concat:output:0flatten_16/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_16/Reshapey
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout_26/dropout/ConstЌ
dropout_26/dropout/MulMuldnn_layer3/Relu:activations:0!dropout_26/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_26/dropout/Mul
dropout_26/dropout/ShapeShapednn_layer3/Relu:activations:0*
T0*
_output_shapes
:2
dropout_26/dropout/Shapeж
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype021
/dropout_26/dropout/random_uniform/RandomUniform
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2#
!dropout_26/dropout/GreaterEqual/yы
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
dropout_26/dropout/GreaterEqualЁ
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout_26/dropout/CastЇ
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_26/dropout/Mul_1Ќ
fm_linear/MatMul/ReadVariableOpReadVariableOp(fm_linear_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
fm_linear/MatMul/ReadVariableOpІ
fm_linear/MatMulMatMulflatten_16/Reshape:output:0'fm_linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_linear/MatMulЊ
 fm_linear/BiasAdd/ReadVariableOpReadVariableOp)fm_linear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 fm_linear/BiasAdd/ReadVariableOpЉ
fm_linear/BiasAddBiasAddfm_linear/MatMul:product:0(fm_linear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_linear/BiasAdd
fm_cross/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
fm_cross/Sum/reduction_indicesК
fm_cross/SumSum%embedding_concatenate/concat:output:0'fm_cross/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
fm_cross/Sumy
fm_cross/SquareSquarefm_cross/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
fm_cross/Square
fm_cross/Square_1Square%embedding_concatenate/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
fm_cross/Square_1
 fm_cross/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 fm_cross/Sum_1/reduction_indicesА
fm_cross/Sum_1Sumfm_cross/Square_1:y:0)fm_cross/Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(2
fm_cross/Sum_1
fm_cross/subSubfm_cross/Square:y:0fm_cross/Sum_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
fm_cross/sub
 fm_cross/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 fm_cross/Sum_2/reduction_indices
fm_cross/Sum_2Sumfm_cross/sub:z:0)fm_cross/Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_cross/Sum_2e
fm_cross/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
fm_cross/mul/x
fm_cross/mulMulfm_cross/mul/x:output:0fm_cross/Sum_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_cross/mul
fm_combine/addAddV2fm_linear/BiasAdd:output:0fm_cross/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
fm_combine/addЏ
 dnn_layer4/MatMul/ReadVariableOpReadVariableOp)dnn_layer4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dnn_layer4/MatMul/ReadVariableOpЊ
dnn_layer4/MatMulMatMuldropout_26/dropout/Mul_1:z:0(dnn_layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dnn_layer4/MatMul
	add_8/addAddV2fm_combine/add:z:0dnn_layer4/MatMul:product:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	add_8/addn
sigmoid/SigmoidSigmoidadd_8/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sigmoid/Sigmoidж
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dnn_layer1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer1/kernel/Regularizer/SquareSquare;dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer1/kernel/Regularizer/Square
#dnn_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer1/kernel/Regularizer/ConstЦ
!dnn_layer1/kernel/Regularizer/SumSum(dnn_layer1/kernel/Regularizer/Square:y:0,dnn_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/Sum
#dnn_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer1/kernel/Regularizer/mul/xШ
!dnn_layer1/kernel/Regularizer/mulMul,dnn_layer1/kernel/Regularizer/mul/x:output:0*dnn_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/mulЮ
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOp*dnn_layer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer1/bias/Regularizer/SquareSquare9dnn_layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer1/bias/Regularizer/Square
!dnn_layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer1/bias/Regularizer/ConstО
dnn_layer1/bias/Regularizer/SumSum&dnn_layer1/bias/Regularizer/Square:y:0*dnn_layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/Sum
!dnn_layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer1/bias/Regularizer/mul/xР
dnn_layer1/bias/Regularizer/mulMul*dnn_layer1/bias/Regularizer/mul/x:output:0(dnn_layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/mulж
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dnn_layer2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer2/kernel/Regularizer/SquareSquare;dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer2/kernel/Regularizer/Square
#dnn_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer2/kernel/Regularizer/ConstЦ
!dnn_layer2/kernel/Regularizer/SumSum(dnn_layer2/kernel/Regularizer/Square:y:0,dnn_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/Sum
#dnn_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer2/kernel/Regularizer/mul/xШ
!dnn_layer2/kernel/Regularizer/mulMul,dnn_layer2/kernel/Regularizer/mul/x:output:0*dnn_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/mulЮ
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOp*dnn_layer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer2/bias/Regularizer/SquareSquare9dnn_layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer2/bias/Regularizer/Square
!dnn_layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer2/bias/Regularizer/ConstО
dnn_layer2/bias/Regularizer/SumSum&dnn_layer2/bias/Regularizer/Square:y:0*dnn_layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/Sum
!dnn_layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer2/bias/Regularizer/mul/xР
dnn_layer2/bias/Regularizer/mulMul*dnn_layer2/bias/Regularizer/mul/x:output:0(dnn_layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/mulж
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dnn_layer3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer3/kernel/Regularizer/SquareSquare;dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer3/kernel/Regularizer/Square
#dnn_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer3/kernel/Regularizer/ConstЦ
!dnn_layer3/kernel/Regularizer/SumSum(dnn_layer3/kernel/Regularizer/Square:y:0,dnn_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/Sum
#dnn_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer3/kernel/Regularizer/mul/xШ
!dnn_layer3/kernel/Regularizer/mulMul,dnn_layer3/kernel/Regularizer/mul/x:output:0*dnn_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/mulЮ
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpReadVariableOp*dnn_layer3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer3/bias/Regularizer/SquareSquare9dnn_layer3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer3/bias/Regularizer/Square
!dnn_layer3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer3/bias/Regularizer/ConstО
dnn_layer3/bias/Regularizer/SumSum&dnn_layer3/bias/Regularizer/Square:y:0*dnn_layer3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/Sum
!dnn_layer3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer3/bias/Regularizer/mul/xР
dnn_layer3/bias/Regularizer/mulMul*dnn_layer3/bias/Regularizer/mul/x:output:0(dnn_layer3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/mulв
2fm_linear/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(fm_linear_matmul_readvariableop_resource*
_output_shapes
:	*
dtype024
2fm_linear/kernel/Regularizer/Square/ReadVariableOpК
#fm_linear/kernel/Regularizer/SquareSquare:fm_linear/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#fm_linear/kernel/Regularizer/Square
"fm_linear/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"fm_linear/kernel/Regularizer/ConstТ
 fm_linear/kernel/Regularizer/SumSum'fm_linear/kernel/Regularizer/Square:y:0+fm_linear/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/Sum
"fm_linear/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2$
"fm_linear/kernel/Regularizer/mul/xФ
 fm_linear/kernel/Regularizer/mulMul+fm_linear/kernel/Regularizer/mul/x:output:0)fm_linear/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/mulЪ
0fm_linear/bias/Regularizer/Square/ReadVariableOpReadVariableOp)fm_linear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0fm_linear/bias/Regularizer/Square/ReadVariableOpЏ
!fm_linear/bias/Regularizer/SquareSquare8fm_linear/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!fm_linear/bias/Regularizer/Square
 fm_linear/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 fm_linear/bias/Regularizer/ConstК
fm_linear/bias/Regularizer/SumSum%fm_linear/bias/Regularizer/Square:y:0)fm_linear/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/Sum
 fm_linear/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2"
 fm_linear/bias/Regularizer/mul/xМ
fm_linear/bias/Regularizer/mulMul)fm_linear/bias/Regularizer/mul/x:output:0'fm_linear/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/mulе
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)dnn_layer4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype025
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpН
$dnn_layer4/kernel/Regularizer/SquareSquare;dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$dnn_layer4/kernel/Regularizer/Square
#dnn_layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer4/kernel/Regularizer/ConstЦ
!dnn_layer4/kernel/Regularizer/SumSum(dnn_layer4/kernel/Regularizer/Square:y:0,dnn_layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/Sum
#dnn_layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer4/kernel/Regularizer/mul/xШ
!dnn_layer4/kernel/Regularizer/mulMul,dnn_layer4/kernel/Regularizer/mul/x:output:0*dnn_layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/mulъ
IdentityIdentitysigmoid/Sigmoid:y:0"^dnn_layer1/BiasAdd/ReadVariableOp!^dnn_layer1/MatMul/ReadVariableOp2^dnn_layer1/bias/Regularizer/Square/ReadVariableOp4^dnn_layer1/kernel/Regularizer/Square/ReadVariableOp"^dnn_layer2/BiasAdd/ReadVariableOp!^dnn_layer2/MatMul/ReadVariableOp2^dnn_layer2/bias/Regularizer/Square/ReadVariableOp4^dnn_layer2/kernel/Regularizer/Square/ReadVariableOp"^dnn_layer3/BiasAdd/ReadVariableOp!^dnn_layer3/MatMul/ReadVariableOp2^dnn_layer3/bias/Regularizer/Square/ReadVariableOp4^dnn_layer3/kernel/Regularizer/Square/ReadVariableOp!^dnn_layer4/MatMul/ReadVariableOp4^dnn_layer4/kernel/Regularizer/Square/ReadVariableOp!^fm_linear/BiasAdd/ReadVariableOp ^fm_linear/MatMul/ReadVariableOp1^fm_linear/bias/Regularizer/Square/ReadVariableOp3^fm_linear/kernel/Regularizer/Square/ReadVariableOp(^item_catalog_embedding/embedding_lookup#^item_id_embedding/embedding_lookup$^item_tag_embedding/embedding_lookup'^member_type_embedding/embedding_lookup#^user_id_embedding/embedding_lookup%^user_type_embedding/embedding_lookup*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::2F
!dnn_layer1/BiasAdd/ReadVariableOp!dnn_layer1/BiasAdd/ReadVariableOp2D
 dnn_layer1/MatMul/ReadVariableOp dnn_layer1/MatMul/ReadVariableOp2f
1dnn_layer1/bias/Regularizer/Square/ReadVariableOp1dnn_layer1/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp2F
!dnn_layer2/BiasAdd/ReadVariableOp!dnn_layer2/BiasAdd/ReadVariableOp2D
 dnn_layer2/MatMul/ReadVariableOp dnn_layer2/MatMul/ReadVariableOp2f
1dnn_layer2/bias/Regularizer/Square/ReadVariableOp1dnn_layer2/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp2F
!dnn_layer3/BiasAdd/ReadVariableOp!dnn_layer3/BiasAdd/ReadVariableOp2D
 dnn_layer3/MatMul/ReadVariableOp dnn_layer3/MatMul/ReadVariableOp2f
1dnn_layer3/bias/Regularizer/Square/ReadVariableOp1dnn_layer3/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp2D
 dnn_layer4/MatMul/ReadVariableOp dnn_layer4/MatMul/ReadVariableOp2j
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp2D
 fm_linear/BiasAdd/ReadVariableOp fm_linear/BiasAdd/ReadVariableOp2B
fm_linear/MatMul/ReadVariableOpfm_linear/MatMul/ReadVariableOp2d
0fm_linear/bias/Regularizer/Square/ReadVariableOp0fm_linear/bias/Regularizer/Square/ReadVariableOp2h
2fm_linear/kernel/Regularizer/Square/ReadVariableOp2fm_linear/kernel/Regularizer/Square/ReadVariableOp2R
'item_catalog_embedding/embedding_lookup'item_catalog_embedding/embedding_lookup2H
"item_id_embedding/embedding_lookup"item_id_embedding/embedding_lookup2J
#item_tag_embedding/embedding_lookup#item_tag_embedding/embedding_lookup2P
&member_type_embedding/embedding_lookup&member_type_embedding/embedding_lookup2H
"user_id_embedding/embedding_lookup"user_id_embedding/embedding_lookup2L
$user_type_embedding/embedding_lookup$user_type_embedding/embedding_lookup:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/5
х

*__inference_dnn_layer2_layer_call_fn_76286

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer2_layer_call_and_return_conditional_losses_748102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
У
p
*__inference_dnn_layer4_layer_call_fn_76496

inputs
unknown
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer4_layer_call_and_return_conditional_losses_750282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
c
*__inference_dropout_26_layer_call_fn_76453

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_26_layer_call_and_return_conditional_losses_749212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
E__inference_dropout_26_layer_call_and_return_conditional_losses_76443

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
хО
Н
C__inference_model_17_layer_call_and_return_conditional_losses_75359

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
user_id_embedding_75253
member_type_embedding_75256
user_type_embedding_75259
item_id_embedding_75262 
item_catalog_embedding_75265
item_tag_embedding_75268
dnn_layer1_75273
dnn_layer1_75275
dnn_layer2_75279
dnn_layer2_75281
dnn_layer3_75285
dnn_layer3_75287
fm_linear_75292
fm_linear_75294
dnn_layer4_75299
identityЂ"dnn_layer1/StatefulPartitionedCallЂ1dnn_layer1/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer2/StatefulPartitionedCallЂ1dnn_layer2/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer3/StatefulPartitionedCallЂ1dnn_layer3/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpЂ"dnn_layer4/StatefulPartitionedCallЂ3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpЂ"dropout_24/StatefulPartitionedCallЂ"dropout_25/StatefulPartitionedCallЂ"dropout_26/StatefulPartitionedCallЂ!fm_linear/StatefulPartitionedCallЂ0fm_linear/bias/Regularizer/Square/ReadVariableOpЂ2fm_linear/kernel/Regularizer/Square/ReadVariableOpЂ.item_catalog_embedding/StatefulPartitionedCallЂ)item_id_embedding/StatefulPartitionedCallЂ*item_tag_embedding/StatefulPartitionedCallЂ-member_type_embedding/StatefulPartitionedCallЂ)user_id_embedding/StatefulPartitionedCallЂ+user_type_embedding/StatefulPartitionedCallЊ
)user_id_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_id_embedding_75253*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_user_id_embedding_layer_call_and_return_conditional_losses_745632+
)user_id_embedding/StatefulPartitionedCallМ
-member_type_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_2member_type_embedding_75256*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_member_type_embedding_layer_call_and_return_conditional_losses_745842/
-member_type_embedding/StatefulPartitionedCallД
+user_type_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1user_type_embedding_75259*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_user_type_embedding_layer_call_and_return_conditional_losses_746052-
+user_type_embedding/StatefulPartitionedCallЌ
)item_id_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_3item_id_embedding_75262*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_item_id_embedding_layer_call_and_return_conditional_losses_746262+
)item_id_embedding/StatefulPartitionedCallР
.item_catalog_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_4item_catalog_embedding_75265*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_item_catalog_embedding_layer_call_and_return_conditional_losses_7464720
.item_catalog_embedding/StatefulPartitionedCallА
*item_tag_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_5item_tag_embedding_75268*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_item_tag_embedding_layer_call_and_return_conditional_losses_746682,
*item_tag_embedding/StatefulPartitionedCallТ
%embedding_concatenate/PartitionedCallPartitionedCall2user_id_embedding/StatefulPartitionedCall:output:06member_type_embedding/StatefulPartitionedCall:output:04user_type_embedding/StatefulPartitionedCall:output:02item_id_embedding/StatefulPartitionedCall:output:07item_catalog_embedding/StatefulPartitionedCall:output:03item_tag_embedding/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_embedding_concatenate_layer_call_and_return_conditional_losses_746912'
%embedding_concatenate/PartitionedCall
flatten_17/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_17_layer_call_and_return_conditional_losses_747102
flatten_17/PartitionedCallМ
"dnn_layer1/StatefulPartitionedCallStatefulPartitionedCall#flatten_17/PartitionedCall:output:0dnn_layer1_75273dnn_layer1_75275*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer1_layer_call_and_return_conditional_losses_747412$
"dnn_layer1/StatefulPartitionedCall
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall+dnn_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_24_layer_call_and_return_conditional_losses_747692$
"dropout_24/StatefulPartitionedCallФ
"dnn_layer2/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0dnn_layer2_75279dnn_layer2_75281*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer2_layer_call_and_return_conditional_losses_748102$
"dnn_layer2/StatefulPartitionedCallП
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall+dnn_layer2/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_25_layer_call_and_return_conditional_losses_748382$
"dropout_25/StatefulPartitionedCallФ
"dnn_layer3/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dnn_layer3_75285dnn_layer3_75287*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer3_layer_call_and_return_conditional_losses_748792$
"dnn_layer3/StatefulPartitionedCall
flatten_16/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_749012
flatten_16/PartitionedCallП
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall+dnn_layer3/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_26_layer_call_and_return_conditional_losses_749212$
"dropout_26/StatefulPartitionedCallЖ
!fm_linear/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0fm_linear_75292fm_linear_75294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_fm_linear_layer_call_and_return_conditional_losses_749612#
!fm_linear/StatefulPartitionedCallў
fm_cross/PartitionedCallPartitionedCall.embedding_concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_fm_cross_layer_call_and_return_conditional_losses_749922
fm_cross/PartitionedCallЄ
fm_combine/PartitionedCallPartitionedCall*fm_linear/StatefulPartitionedCall:output:0!fm_cross/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_fm_combine_layer_call_and_return_conditional_losses_750062
fm_combine/PartitionedCallЏ
"dnn_layer4/StatefulPartitionedCallStatefulPartitionedCall+dropout_26/StatefulPartitionedCall:output:0dnn_layer4_75299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer4_layer_call_and_return_conditional_losses_750282$
"dnn_layer4/StatefulPartitionedCall
add_8/PartitionedCallPartitionedCall#fm_combine/PartitionedCall:output:0+dnn_layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_8_layer_call_and_return_conditional_losses_750462
add_8/PartitionedCallы
sigmoid/PartitionedCallPartitionedCalladd_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_sigmoid_layer_call_and_return_conditional_losses_750602
sigmoid/PartitionedCallН
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer1_75273* 
_output_shapes
:
*
dtype025
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer1/kernel/Regularizer/SquareSquare;dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer1/kernel/Regularizer/Square
#dnn_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer1/kernel/Regularizer/ConstЦ
!dnn_layer1/kernel/Regularizer/SumSum(dnn_layer1/kernel/Regularizer/Square:y:0,dnn_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/Sum
#dnn_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer1/kernel/Regularizer/mul/xШ
!dnn_layer1/kernel/Regularizer/mulMul,dnn_layer1/kernel/Regularizer/mul/x:output:0*dnn_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/mulД
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer1_75275*
_output_shapes	
:*
dtype023
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer1/bias/Regularizer/SquareSquare9dnn_layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer1/bias/Regularizer/Square
!dnn_layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer1/bias/Regularizer/ConstО
dnn_layer1/bias/Regularizer/SumSum&dnn_layer1/bias/Regularizer/Square:y:0*dnn_layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/Sum
!dnn_layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer1/bias/Regularizer/mul/xР
dnn_layer1/bias/Regularizer/mulMul*dnn_layer1/bias/Regularizer/mul/x:output:0(dnn_layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/mulН
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer2_75279* 
_output_shapes
:
*
dtype025
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer2/kernel/Regularizer/SquareSquare;dnn_layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer2/kernel/Regularizer/Square
#dnn_layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer2/kernel/Regularizer/ConstЦ
!dnn_layer2/kernel/Regularizer/SumSum(dnn_layer2/kernel/Regularizer/Square:y:0,dnn_layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/Sum
#dnn_layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer2/kernel/Regularizer/mul/xШ
!dnn_layer2/kernel/Regularizer/mulMul,dnn_layer2/kernel/Regularizer/mul/x:output:0*dnn_layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer2/kernel/Regularizer/mulД
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer2_75281*
_output_shapes	
:*
dtype023
1dnn_layer2/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer2/bias/Regularizer/SquareSquare9dnn_layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer2/bias/Regularizer/Square
!dnn_layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer2/bias/Regularizer/ConstО
dnn_layer2/bias/Regularizer/SumSum&dnn_layer2/bias/Regularizer/Square:y:0*dnn_layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/Sum
!dnn_layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer2/bias/Regularizer/mul/xР
dnn_layer2/bias/Regularizer/mulMul*dnn_layer2/bias/Regularizer/mul/x:output:0(dnn_layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer2/bias/Regularizer/mulН
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer3_75285* 
_output_shapes
:
*
dtype025
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer3/kernel/Regularizer/SquareSquare;dnn_layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer3/kernel/Regularizer/Square
#dnn_layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer3/kernel/Regularizer/ConstЦ
!dnn_layer3/kernel/Regularizer/SumSum(dnn_layer3/kernel/Regularizer/Square:y:0,dnn_layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/Sum
#dnn_layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer3/kernel/Regularizer/mul/xШ
!dnn_layer3/kernel/Regularizer/mulMul,dnn_layer3/kernel/Regularizer/mul/x:output:0*dnn_layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer3/kernel/Regularizer/mulД
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer3_75287*
_output_shapes	
:*
dtype023
1dnn_layer3/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer3/bias/Regularizer/SquareSquare9dnn_layer3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer3/bias/Regularizer/Square
!dnn_layer3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer3/bias/Regularizer/ConstО
dnn_layer3/bias/Regularizer/SumSum&dnn_layer3/bias/Regularizer/Square:y:0*dnn_layer3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/Sum
!dnn_layer3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer3/bias/Regularizer/mul/xР
dnn_layer3/bias/Regularizer/mulMul*dnn_layer3/bias/Regularizer/mul/x:output:0(dnn_layer3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer3/bias/Regularizer/mulЙ
2fm_linear/kernel/Regularizer/Square/ReadVariableOpReadVariableOpfm_linear_75292*
_output_shapes
:	*
dtype024
2fm_linear/kernel/Regularizer/Square/ReadVariableOpК
#fm_linear/kernel/Regularizer/SquareSquare:fm_linear/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#fm_linear/kernel/Regularizer/Square
"fm_linear/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"fm_linear/kernel/Regularizer/ConstТ
 fm_linear/kernel/Regularizer/SumSum'fm_linear/kernel/Regularizer/Square:y:0+fm_linear/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/Sum
"fm_linear/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2$
"fm_linear/kernel/Regularizer/mul/xФ
 fm_linear/kernel/Regularizer/mulMul+fm_linear/kernel/Regularizer/mul/x:output:0)fm_linear/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/mulА
0fm_linear/bias/Regularizer/Square/ReadVariableOpReadVariableOpfm_linear_75294*
_output_shapes
:*
dtype022
0fm_linear/bias/Regularizer/Square/ReadVariableOpЏ
!fm_linear/bias/Regularizer/SquareSquare8fm_linear/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!fm_linear/bias/Regularizer/Square
 fm_linear/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 fm_linear/bias/Regularizer/ConstК
fm_linear/bias/Regularizer/SumSum%fm_linear/bias/Regularizer/Square:y:0)fm_linear/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/Sum
 fm_linear/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2"
 fm_linear/bias/Regularizer/mul/xМ
fm_linear/bias/Regularizer/mulMul)fm_linear/bias/Regularizer/mul/x:output:0'fm_linear/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/mulМ
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdnn_layer4_75299*
_output_shapes
:	*
dtype025
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpН
$dnn_layer4/kernel/Regularizer/SquareSquare;dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$dnn_layer4/kernel/Regularizer/Square
#dnn_layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer4/kernel/Regularizer/ConstЦ
!dnn_layer4/kernel/Regularizer/SumSum(dnn_layer4/kernel/Regularizer/Square:y:0,dnn_layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/Sum
#dnn_layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer4/kernel/Regularizer/mul/xШ
!dnn_layer4/kernel/Regularizer/mulMul,dnn_layer4/kernel/Regularizer/mul/x:output:0*dnn_layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/mul	
IdentityIdentity sigmoid/PartitionedCall:output:0#^dnn_layer1/StatefulPartitionedCall2^dnn_layer1/bias/Regularizer/Square/ReadVariableOp4^dnn_layer1/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer2/StatefulPartitionedCall2^dnn_layer2/bias/Regularizer/Square/ReadVariableOp4^dnn_layer2/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer3/StatefulPartitionedCall2^dnn_layer3/bias/Regularizer/Square/ReadVariableOp4^dnn_layer3/kernel/Regularizer/Square/ReadVariableOp#^dnn_layer4/StatefulPartitionedCall4^dnn_layer4/kernel/Regularizer/Square/ReadVariableOp#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall"^fm_linear/StatefulPartitionedCall1^fm_linear/bias/Regularizer/Square/ReadVariableOp3^fm_linear/kernel/Regularizer/Square/ReadVariableOp/^item_catalog_embedding/StatefulPartitionedCall*^item_id_embedding/StatefulPartitionedCall+^item_tag_embedding/StatefulPartitionedCall.^member_type_embedding/StatefulPartitionedCall*^user_id_embedding/StatefulPartitionedCall,^user_type_embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::2H
"dnn_layer1/StatefulPartitionedCall"dnn_layer1/StatefulPartitionedCall2f
1dnn_layer1/bias/Regularizer/Square/ReadVariableOp1dnn_layer1/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer2/StatefulPartitionedCall"dnn_layer2/StatefulPartitionedCall2f
1dnn_layer2/bias/Regularizer/Square/ReadVariableOp1dnn_layer2/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp3dnn_layer2/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer3/StatefulPartitionedCall"dnn_layer3/StatefulPartitionedCall2f
1dnn_layer3/bias/Regularizer/Square/ReadVariableOp1dnn_layer3/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp3dnn_layer3/kernel/Regularizer/Square/ReadVariableOp2H
"dnn_layer4/StatefulPartitionedCall"dnn_layer4/StatefulPartitionedCall2j
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall2F
!fm_linear/StatefulPartitionedCall!fm_linear/StatefulPartitionedCall2d
0fm_linear/bias/Regularizer/Square/ReadVariableOp0fm_linear/bias/Regularizer/Square/ReadVariableOp2h
2fm_linear/kernel/Regularizer/Square/ReadVariableOp2fm_linear/kernel/Regularizer/Square/ReadVariableOp2`
.item_catalog_embedding/StatefulPartitionedCall.item_catalog_embedding/StatefulPartitionedCall2V
)item_id_embedding/StatefulPartitionedCall)item_id_embedding/StatefulPartitionedCall2X
*item_tag_embedding/StatefulPartitionedCall*item_tag_embedding/StatefulPartitionedCall2^
-member_type_embedding/StatefulPartitionedCall-member_type_embedding/StatefulPartitionedCall2V
)user_id_embedding/StatefulPartitionedCall)user_id_embedding/StatefulPartitionedCall2Z
+user_type_embedding/StatefulPartitionedCall+user_type_embedding/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Х
D__inference_fm_linear_layer_call_and_return_conditional_losses_74961

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ0fm_linear/bias/Regularizer/Square/ReadVariableOpЂ2fm_linear/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddШ
2fm_linear/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype024
2fm_linear/kernel/Regularizer/Square/ReadVariableOpК
#fm_linear/kernel/Regularizer/SquareSquare:fm_linear/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#fm_linear/kernel/Regularizer/Square
"fm_linear/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"fm_linear/kernel/Regularizer/ConstТ
 fm_linear/kernel/Regularizer/SumSum'fm_linear/kernel/Regularizer/Square:y:0+fm_linear/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/Sum
"fm_linear/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2$
"fm_linear/kernel/Regularizer/mul/xФ
 fm_linear/kernel/Regularizer/mulMul+fm_linear/kernel/Regularizer/mul/x:output:0)fm_linear/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/mulР
0fm_linear/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0fm_linear/bias/Regularizer/Square/ReadVariableOpЏ
!fm_linear/bias/Regularizer/SquareSquare8fm_linear/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!fm_linear/bias/Regularizer/Square
 fm_linear/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 fm_linear/bias/Regularizer/ConstК
fm_linear/bias/Regularizer/SumSum%fm_linear/bias/Regularizer/Square:y:0)fm_linear/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/Sum
 fm_linear/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2"
 fm_linear/bias/Regularizer/mul/xМ
fm_linear/bias/Regularizer/mulMul)fm_linear/bias/Regularizer/mul/x:output:0'fm_linear/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/mul§
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^fm_linear/bias/Regularizer/Square/ReadVariableOp3^fm_linear/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0fm_linear/bias/Regularizer/Square/ReadVariableOp0fm_linear/bias/Regularizer/Square/ReadVariableOp2h
2fm_linear/kernel/Regularizer/Square/ReadVariableOp2fm_linear/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ь
c
E__inference_dropout_25_layer_call_and_return_conditional_losses_76303

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§ј
Ы 
!__inference__traced_restore_77003
file_prefix1
-assignvariableop_user_id_embedding_embeddings7
3assignvariableop_1_member_type_embedding_embeddings5
1assignvariableop_2_user_type_embedding_embeddings3
/assignvariableop_3_item_id_embedding_embeddings8
4assignvariableop_4_item_catalog_embedding_embeddings4
0assignvariableop_5_item_tag_embedding_embeddings(
$assignvariableop_6_dnn_layer1_kernel&
"assignvariableop_7_dnn_layer1_bias(
$assignvariableop_8_dnn_layer2_kernel&
"assignvariableop_9_dnn_layer2_bias)
%assignvariableop_10_dnn_layer3_kernel'
#assignvariableop_11_dnn_layer3_bias(
$assignvariableop_12_fm_linear_kernel&
"assignvariableop_13_fm_linear_bias)
%assignvariableop_14_dnn_layer4_kernel!
assignvariableop_15_adam_iter#
assignvariableop_16_adam_beta_1#
assignvariableop_17_adam_beta_2"
assignvariableop_18_adam_decay*
&assignvariableop_19_adam_learning_rate
assignvariableop_20_total
assignvariableop_21_count&
"assignvariableop_22_true_positives&
"assignvariableop_23_true_negatives'
#assignvariableop_24_false_positives'
#assignvariableop_25_false_negatives(
$assignvariableop_26_true_positives_1)
%assignvariableop_27_false_negatives_1;
7assignvariableop_28_adam_user_id_embedding_embeddings_m?
;assignvariableop_29_adam_member_type_embedding_embeddings_m=
9assignvariableop_30_adam_user_type_embedding_embeddings_m;
7assignvariableop_31_adam_item_id_embedding_embeddings_m@
<assignvariableop_32_adam_item_catalog_embedding_embeddings_m<
8assignvariableop_33_adam_item_tag_embedding_embeddings_m0
,assignvariableop_34_adam_dnn_layer1_kernel_m.
*assignvariableop_35_adam_dnn_layer1_bias_m0
,assignvariableop_36_adam_dnn_layer2_kernel_m.
*assignvariableop_37_adam_dnn_layer2_bias_m0
,assignvariableop_38_adam_dnn_layer3_kernel_m.
*assignvariableop_39_adam_dnn_layer3_bias_m/
+assignvariableop_40_adam_fm_linear_kernel_m-
)assignvariableop_41_adam_fm_linear_bias_m0
,assignvariableop_42_adam_dnn_layer4_kernel_m;
7assignvariableop_43_adam_user_id_embedding_embeddings_v?
;assignvariableop_44_adam_member_type_embedding_embeddings_v=
9assignvariableop_45_adam_user_type_embedding_embeddings_v;
7assignvariableop_46_adam_item_id_embedding_embeddings_v@
<assignvariableop_47_adam_item_catalog_embedding_embeddings_v<
8assignvariableop_48_adam_item_tag_embedding_embeddings_v0
,assignvariableop_49_adam_dnn_layer1_kernel_v.
*assignvariableop_50_adam_dnn_layer1_bias_v0
,assignvariableop_51_adam_dnn_layer2_kernel_v.
*assignvariableop_52_adam_dnn_layer2_bias_v0
,assignvariableop_53_adam_dnn_layer3_kernel_v.
*assignvariableop_54_adam_dnn_layer3_bias_v/
+assignvariableop_55_adam_fm_linear_kernel_v-
)assignvariableop_56_adam_fm_linear_bias_v0
,assignvariableop_57_adam_dnn_layer4_kernel_v
identity_59ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ш!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*д 
valueЪ BЧ ;B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*
valueB~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesе
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesя
ь:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЌ
AssignVariableOpAssignVariableOp-assignvariableop_user_id_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1И
AssignVariableOp_1AssignVariableOp3assignvariableop_1_member_type_embedding_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ж
AssignVariableOp_2AssignVariableOp1assignvariableop_2_user_type_embedding_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Д
AssignVariableOp_3AssignVariableOp/assignvariableop_3_item_id_embedding_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Й
AssignVariableOp_4AssignVariableOp4assignvariableop_4_item_catalog_embedding_embeddingsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Е
AssignVariableOp_5AssignVariableOp0assignvariableop_5_item_tag_embedding_embeddingsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Љ
AssignVariableOp_6AssignVariableOp$assignvariableop_6_dnn_layer1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ї
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dnn_layer1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Љ
AssignVariableOp_8AssignVariableOp$assignvariableop_8_dnn_layer2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ї
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dnn_layer2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10­
AssignVariableOp_10AssignVariableOp%assignvariableop_10_dnn_layer3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ћ
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dnn_layer3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ќ
AssignVariableOp_12AssignVariableOp$assignvariableop_12_fm_linear_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Њ
AssignVariableOp_13AssignVariableOp"assignvariableop_13_fm_linear_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14­
AssignVariableOp_14AssignVariableOp%assignvariableop_14_dnn_layer4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_15Ѕ
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_iterIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ї
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ї
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18І
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ў
AssignVariableOp_19AssignVariableOp&assignvariableop_19_adam_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ё
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ё
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Њ
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_positivesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Њ
AssignVariableOp_23AssignVariableOp"assignvariableop_23_true_negativesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ћ
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_positivesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ћ
AssignVariableOp_25AssignVariableOp#assignvariableop_25_false_negativesIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ќ
AssignVariableOp_26AssignVariableOp$assignvariableop_26_true_positives_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27­
AssignVariableOp_27AssignVariableOp%assignvariableop_27_false_negatives_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28П
AssignVariableOp_28AssignVariableOp7assignvariableop_28_adam_user_id_embedding_embeddings_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29У
AssignVariableOp_29AssignVariableOp;assignvariableop_29_adam_member_type_embedding_embeddings_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30С
AssignVariableOp_30AssignVariableOp9assignvariableop_30_adam_user_type_embedding_embeddings_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31П
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adam_item_id_embedding_embeddings_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ф
AssignVariableOp_32AssignVariableOp<assignvariableop_32_adam_item_catalog_embedding_embeddings_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Р
AssignVariableOp_33AssignVariableOp8assignvariableop_33_adam_item_tag_embedding_embeddings_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Д
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_dnn_layer1_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35В
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dnn_layer1_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Д
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_dnn_layer2_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37В
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dnn_layer2_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Д
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_dnn_layer3_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39В
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dnn_layer3_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Г
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_fm_linear_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Б
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_fm_linear_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Д
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_dnn_layer4_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43П
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_user_id_embedding_embeddings_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44У
AssignVariableOp_44AssignVariableOp;assignvariableop_44_adam_member_type_embedding_embeddings_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45С
AssignVariableOp_45AssignVariableOp9assignvariableop_45_adam_user_type_embedding_embeddings_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46П
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adam_item_id_embedding_embeddings_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ф
AssignVariableOp_47AssignVariableOp<assignvariableop_47_adam_item_catalog_embedding_embeddings_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Р
AssignVariableOp_48AssignVariableOp8assignvariableop_48_adam_item_tag_embedding_embeddings_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Д
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_dnn_layer1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50В
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dnn_layer1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Д
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_dnn_layer2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52В
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_dnn_layer2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Д
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_dnn_layer3_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54В
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_dnn_layer3_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Г
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_fm_linear_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Б
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_fm_linear_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Д
AssignVariableOp_57AssignVariableOp,assignvariableop_57_adam_dnn_layer4_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_579
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpк

Identity_58Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_58Э

Identity_59IdentityIdentity_58:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_59"#
identity_59Identity_59:output:0*џ
_input_shapesэ
ъ: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ъ


5__inference_embedding_concatenate_layer_call_fn_76160
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_embedding_concatenate_layer_call_and_return_conditional_losses_746912
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/3:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/4:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/5

Q
%__inference_add_8_layer_call_fn_76508
inputs_0
inputs_1
identityЮ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_8_layer_call_and_return_conditional_losses_750462
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
п
{
5__inference_member_type_embedding_layer_call_fn_76075

inputs
unknown
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_member_type_embedding_layer_call_and_return_conditional_losses_745842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
^
B__inference_sigmoid_layer_call_and_return_conditional_losses_75060

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


(__inference_model_17_layer_call_fn_76003
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_17_layer_call_and_return_conditional_losses_753592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/5


(__inference_model_17_layer_call_fn_75392
user_id
	user_type
member_type
item_id
item_catalog
item_tag
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCalluser_id	user_typemember_typeitem_iditem_catalogitem_tagunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_17_layer_call_and_return_conditional_losses_753592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*У
_input_shapesБ
Ў:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	user_id:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	user_type:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_namemember_type:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	item_id:UQ
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameitem_catalog:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
item_tag

d
E__inference_dropout_26_layer_call_and_return_conditional_losses_74921

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	

N__inference_user_type_embedding_layer_call_and_return_conditional_losses_76084

inputs
embedding_lookup_76078
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_76078inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/76078*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/76078*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ	
Д
P__inference_embedding_concatenate_layer_call_and_return_conditional_losses_76150
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis­
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:U Q
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/3:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/4:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/5
Ь
c
E__inference_dropout_24_layer_call_and_return_conditional_losses_74774

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№y
 
__inference__traced_save_76819
file_prefix;
7savev2_user_id_embedding_embeddings_read_readvariableop?
;savev2_member_type_embedding_embeddings_read_readvariableop=
9savev2_user_type_embedding_embeddings_read_readvariableop;
7savev2_item_id_embedding_embeddings_read_readvariableop@
<savev2_item_catalog_embedding_embeddings_read_readvariableop<
8savev2_item_tag_embedding_embeddings_read_readvariableop0
,savev2_dnn_layer1_kernel_read_readvariableop.
*savev2_dnn_layer1_bias_read_readvariableop0
,savev2_dnn_layer2_kernel_read_readvariableop.
*savev2_dnn_layer2_bias_read_readvariableop0
,savev2_dnn_layer3_kernel_read_readvariableop.
*savev2_dnn_layer3_bias_read_readvariableop/
+savev2_fm_linear_kernel_read_readvariableop-
)savev2_fm_linear_bias_read_readvariableop0
,savev2_dnn_layer4_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableopB
>savev2_adam_user_id_embedding_embeddings_m_read_readvariableopF
Bsavev2_adam_member_type_embedding_embeddings_m_read_readvariableopD
@savev2_adam_user_type_embedding_embeddings_m_read_readvariableopB
>savev2_adam_item_id_embedding_embeddings_m_read_readvariableopG
Csavev2_adam_item_catalog_embedding_embeddings_m_read_readvariableopC
?savev2_adam_item_tag_embedding_embeddings_m_read_readvariableop7
3savev2_adam_dnn_layer1_kernel_m_read_readvariableop5
1savev2_adam_dnn_layer1_bias_m_read_readvariableop7
3savev2_adam_dnn_layer2_kernel_m_read_readvariableop5
1savev2_adam_dnn_layer2_bias_m_read_readvariableop7
3savev2_adam_dnn_layer3_kernel_m_read_readvariableop5
1savev2_adam_dnn_layer3_bias_m_read_readvariableop6
2savev2_adam_fm_linear_kernel_m_read_readvariableop4
0savev2_adam_fm_linear_bias_m_read_readvariableop7
3savev2_adam_dnn_layer4_kernel_m_read_readvariableopB
>savev2_adam_user_id_embedding_embeddings_v_read_readvariableopF
Bsavev2_adam_member_type_embedding_embeddings_v_read_readvariableopD
@savev2_adam_user_type_embedding_embeddings_v_read_readvariableopB
>savev2_adam_item_id_embedding_embeddings_v_read_readvariableopG
Csavev2_adam_item_catalog_embedding_embeddings_v_read_readvariableopC
?savev2_adam_item_tag_embedding_embeddings_v_read_readvariableop7
3savev2_adam_dnn_layer1_kernel_v_read_readvariableop5
1savev2_adam_dnn_layer1_bias_v_read_readvariableop7
3savev2_adam_dnn_layer2_kernel_v_read_readvariableop5
1savev2_adam_dnn_layer2_bias_v_read_readvariableop7
3savev2_adam_dnn_layer3_kernel_v_read_readvariableop5
1savev2_adam_dnn_layer3_bias_v_read_readvariableop6
2savev2_adam_fm_linear_kernel_v_read_readvariableop4
0savev2_adam_fm_linear_bias_v_read_readvariableop7
3savev2_adam_dnn_layer4_kernel_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameТ!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*д 
valueЪ BЧ ;B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*
valueB~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesС
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_user_id_embedding_embeddings_read_readvariableop;savev2_member_type_embedding_embeddings_read_readvariableop9savev2_user_type_embedding_embeddings_read_readvariableop7savev2_item_id_embedding_embeddings_read_readvariableop<savev2_item_catalog_embedding_embeddings_read_readvariableop8savev2_item_tag_embedding_embeddings_read_readvariableop,savev2_dnn_layer1_kernel_read_readvariableop*savev2_dnn_layer1_bias_read_readvariableop,savev2_dnn_layer2_kernel_read_readvariableop*savev2_dnn_layer2_bias_read_readvariableop,savev2_dnn_layer3_kernel_read_readvariableop*savev2_dnn_layer3_bias_read_readvariableop+savev2_fm_linear_kernel_read_readvariableop)savev2_fm_linear_bias_read_readvariableop,savev2_dnn_layer4_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop>savev2_adam_user_id_embedding_embeddings_m_read_readvariableopBsavev2_adam_member_type_embedding_embeddings_m_read_readvariableop@savev2_adam_user_type_embedding_embeddings_m_read_readvariableop>savev2_adam_item_id_embedding_embeddings_m_read_readvariableopCsavev2_adam_item_catalog_embedding_embeddings_m_read_readvariableop?savev2_adam_item_tag_embedding_embeddings_m_read_readvariableop3savev2_adam_dnn_layer1_kernel_m_read_readvariableop1savev2_adam_dnn_layer1_bias_m_read_readvariableop3savev2_adam_dnn_layer2_kernel_m_read_readvariableop1savev2_adam_dnn_layer2_bias_m_read_readvariableop3savev2_adam_dnn_layer3_kernel_m_read_readvariableop1savev2_adam_dnn_layer3_bias_m_read_readvariableop2savev2_adam_fm_linear_kernel_m_read_readvariableop0savev2_adam_fm_linear_bias_m_read_readvariableop3savev2_adam_dnn_layer4_kernel_m_read_readvariableop>savev2_adam_user_id_embedding_embeddings_v_read_readvariableopBsavev2_adam_member_type_embedding_embeddings_v_read_readvariableop@savev2_adam_user_type_embedding_embeddings_v_read_readvariableop>savev2_adam_item_id_embedding_embeddings_v_read_readvariableopCsavev2_adam_item_catalog_embedding_embeddings_v_read_readvariableop?savev2_adam_item_tag_embedding_embeddings_v_read_readvariableop3savev2_adam_dnn_layer1_kernel_v_read_readvariableop1savev2_adam_dnn_layer1_bias_v_read_readvariableop3savev2_adam_dnn_layer2_kernel_v_read_readvariableop1savev2_adam_dnn_layer2_bias_v_read_readvariableop3savev2_adam_dnn_layer3_kernel_v_read_readvariableop1savev2_adam_dnn_layer3_bias_v_read_readvariableop2savev2_adam_fm_linear_kernel_v_read_readvariableop0savev2_adam_fm_linear_bias_v_read_readvariableop3savev2_adam_dnn_layer4_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *I
dtypes?
=2;	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*
_input_shapesќ
љ: :
рд:::	Ќ::	­:
::
::
::	::	: : : : : : : :Ш:Ш:Ш:Ш:::
рд:::	Ќ::	­:
::
::
::	::	:
рд:::	Ќ::	­:
::
::
::	::	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
рд:$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	Ќ:$ 

_output_shapes

::%!

_output_shapes
:	­:&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:Ш:!

_output_shapes	
:Ш:!

_output_shapes	
:Ш:!

_output_shapes	
:Ш: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
рд:$ 

_output_shapes

::$ 

_output_shapes

::% !

_output_shapes
:	Ќ:$! 

_output_shapes

::%"!

_output_shapes
:	­:&#"
 
_output_shapes
:
:!$

_output_shapes	
::&%"
 
_output_shapes
:
:!&

_output_shapes	
::&'"
 
_output_shapes
:
:!(

_output_shapes	
::%)!

_output_shapes
:	: *

_output_shapes
::%+!

_output_shapes
:	:&,"
 
_output_shapes
:
рд:$- 

_output_shapes

::$. 

_output_shapes

::%/!

_output_shapes
:	Ќ:$0 

_output_shapes

::%1!

_output_shapes
:	­:&2"
 
_output_shapes
:
:!3

_output_shapes	
::&4"
 
_output_shapes
:
:!5

_output_shapes	
::&6"
 
_output_shapes
:
:!7

_output_shapes	
::%8!

_output_shapes
:	: 9

_output_shapes
::%:!

_output_shapes
:	:;

_output_shapes
: 
	

P__inference_member_type_embedding_layer_call_and_return_conditional_losses_74584

inputs
embedding_lookup_74578
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_74578inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/74578*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/74578*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х

*__inference_dnn_layer3_layer_call_fn_76368

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dnn_layer3_layer_call_and_return_conditional_losses_748792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Х
D__inference_fm_linear_layer_call_and_return_conditional_losses_76402

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ0fm_linear/bias/Regularizer/Square/ReadVariableOpЂ2fm_linear/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddШ
2fm_linear/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype024
2fm_linear/kernel/Regularizer/Square/ReadVariableOpК
#fm_linear/kernel/Regularizer/SquareSquare:fm_linear/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2%
#fm_linear/kernel/Regularizer/Square
"fm_linear/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"fm_linear/kernel/Regularizer/ConstТ
 fm_linear/kernel/Regularizer/SumSum'fm_linear/kernel/Regularizer/Square:y:0+fm_linear/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/Sum
"fm_linear/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2$
"fm_linear/kernel/Regularizer/mul/xФ
 fm_linear/kernel/Regularizer/mulMul+fm_linear/kernel/Regularizer/mul/x:output:0)fm_linear/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 fm_linear/kernel/Regularizer/mulР
0fm_linear/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0fm_linear/bias/Regularizer/Square/ReadVariableOpЏ
!fm_linear/bias/Regularizer/SquareSquare8fm_linear/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!fm_linear/bias/Regularizer/Square
 fm_linear/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 fm_linear/bias/Regularizer/ConstК
fm_linear/bias/Regularizer/SumSum%fm_linear/bias/Regularizer/Square:y:0)fm_linear/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/Sum
 fm_linear/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2"
 fm_linear/bias/Regularizer/mul/xМ
fm_linear/bias/Regularizer/mulMul)fm_linear/bias/Regularizer/mul/x:output:0'fm_linear/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/mul§
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^fm_linear/bias/Regularizer/Square/ReadVariableOp3^fm_linear/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0fm_linear/bias/Regularizer/Square/ReadVariableOp0fm_linear/bias/Regularizer/Square/ReadVariableOp2h
2fm_linear/kernel/Regularizer/Square/ReadVariableOp2fm_linear/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

F
*__inference_dropout_26_layer_call_fn_76458

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_26_layer_call_and_return_conditional_losses_749262
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ш
E__inference_dnn_layer1_layer_call_and_return_conditional_losses_76206

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ1dnn_layer1/bias/Regularizer/Square/ReadVariableOpЂ3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ReluЫ
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOpО
$dnn_layer1/kernel/Regularizer/SquareSquare;dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2&
$dnn_layer1/kernel/Regularizer/Square
#dnn_layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer1/kernel/Regularizer/ConstЦ
!dnn_layer1/kernel/Regularizer/SumSum(dnn_layer1/kernel/Regularizer/Square:y:0,dnn_layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/Sum
#dnn_layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer1/kernel/Regularizer/mul/xШ
!dnn_layer1/kernel/Regularizer/mulMul,dnn_layer1/kernel/Regularizer/mul/x:output:0*dnn_layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer1/kernel/Regularizer/mulУ
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1dnn_layer1/bias/Regularizer/Square/ReadVariableOpГ
"dnn_layer1/bias/Regularizer/SquareSquare9dnn_layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"dnn_layer1/bias/Regularizer/Square
!dnn_layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dnn_layer1/bias/Regularizer/ConstО
dnn_layer1/bias/Regularizer/SumSum&dnn_layer1/bias/Regularizer/Square:y:0*dnn_layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/Sum
!dnn_layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2#
!dnn_layer1/bias/Regularizer/mul/xР
dnn_layer1/bias/Regularizer/mulMul*dnn_layer1/bias/Regularizer/mul/x:output:0(dnn_layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dnn_layer1/bias/Regularizer/mul
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dnn_layer1/bias/Regularizer/Square/ReadVariableOp4^dnn_layer1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dnn_layer1/bias/Regularizer/Square/ReadVariableOp1dnn_layer1/bias/Regularizer/Square/ReadVariableOp2j
3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp3dnn_layer1/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
н
І
__inference_loss_fn_8_76617@
<dnn_layer4_kernel_regularizer_square_readvariableop_resource
identityЂ3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpш
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<dnn_layer4_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype025
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOpН
$dnn_layer4/kernel/Regularizer/SquareSquare;dnn_layer4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$dnn_layer4/kernel/Regularizer/Square
#dnn_layer4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#dnn_layer4/kernel/Regularizer/ConstЦ
!dnn_layer4/kernel/Regularizer/SumSum(dnn_layer4/kernel/Regularizer/Square:y:0,dnn_layer4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/Sum
#dnn_layer4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2%
#dnn_layer4/kernel/Regularizer/mul/xШ
!dnn_layer4/kernel/Regularizer/mulMul,dnn_layer4/kernel/Regularizer/mul/x:output:0*dnn_layer4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dnn_layer4/kernel/Regularizer/mul
IdentityIdentity%dnn_layer4/kernel/Regularizer/mul:z:04^dnn_layer4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2j
3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp3dnn_layer4/kernel/Regularizer/Square/ReadVariableOp
з
w
1__inference_user_id_embedding_layer_call_fn_76059

inputs
unknown
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_user_id_embedding_layer_call_and_return_conditional_losses_745632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	

Q__inference_item_catalog_embedding_layer_call_and_return_conditional_losses_76116

inputs
embedding_lookup_76110
identityЂembedding_lookupљ
embedding_lookupResourceGatherembedding_lookup_76110inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/76110*+
_output_shapes
:џџџџџџџџџ*
dtype02
embedding_lookupь
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/76110*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
й
x
2__inference_item_tag_embedding_layer_call_fn_76139

inputs
unknown
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_item_tag_embedding_layer_call_and_return_conditional_losses_746682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

 
__inference_loss_fn_7_76606=
9fm_linear_bias_regularizer_square_readvariableop_resource
identityЂ0fm_linear/bias/Regularizer/Square/ReadVariableOpк
0fm_linear/bias/Regularizer/Square/ReadVariableOpReadVariableOp9fm_linear_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype022
0fm_linear/bias/Regularizer/Square/ReadVariableOpЏ
!fm_linear/bias/Regularizer/SquareSquare8fm_linear/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!fm_linear/bias/Regularizer/Square
 fm_linear/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 fm_linear/bias/Regularizer/ConstК
fm_linear/bias/Regularizer/SumSum%fm_linear/bias/Regularizer/Square:y:0)fm_linear/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/Sum
 fm_linear/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL@2"
 fm_linear/bias/Regularizer/mul/xМ
fm_linear/bias/Regularizer/mulMul)fm_linear/bias/Regularizer/mul/x:output:0'fm_linear/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
fm_linear/bias/Regularizer/mul
IdentityIdentity"fm_linear/bias/Regularizer/mul:z:01^fm_linear/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2d
0fm_linear/bias/Regularizer/Square/ReadVariableOp0fm_linear/bias/Regularizer/Square/ReadVariableOp"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ѓ
serving_defaultп
E
item_catalog5
serving_default_item_catalog:0џџџџџџџџџ
;
item_id0
serving_default_item_id:0џџџџџџџџџ
=
item_tag1
serving_default_item_tag:0џџџџџџџџџ
C
member_type4
serving_default_member_type:0џџџџџџџџџ
;
user_id0
serving_default_user_id:0џџџџџџџџџ
?
	user_type2
serving_default_user_type:0џџџџџџџџџ;
sigmoid0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:к
Јд
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer_with_weights-9
layer-20
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer-25
layer-26
	optimizer
regularization_losses
	variables
trainable_variables
 	keras_api
!
signatures
Ј_default_save_signature
+Љ&call_and_return_all_conditional_losses
Њ__call__"ЪЭ
_tf_keras_network­Э{"class_name": "Functional", "name": "model_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "user_id"}, "name": "user_id", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "member_type"}, "name": "member_type", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "user_type"}, "name": "user_type", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_id"}, "name": "item_id", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_catalog"}, "name": "item_catalog", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_tag"}, "name": "item_tag", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "user_id_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 60000, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "user_id_embedding", "inbound_nodes": [[["user_id", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "member_type_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 5, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "member_type_embedding", "inbound_nodes": [[["member_type", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "user_type_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "user_type_embedding", "inbound_nodes": [[["user_type", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "item_id_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 300, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "item_id_embedding", "inbound_nodes": [[["item_id", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "item_catalog_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 14, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "item_catalog_embedding", "inbound_nodes": [[["item_catalog", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "item_tag_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 301, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "item_tag_embedding", "inbound_nodes": [[["item_tag", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "embedding_concatenate", "trainable": true, "dtype": "float32", "axis": 1}, "name": "embedding_concatenate", "inbound_nodes": [[["user_id_embedding", 0, 0, {}], ["member_type_embedding", 0, 0, {}], ["user_type_embedding", 0, 0, {}], ["item_id_embedding", 0, 0, {}], ["item_catalog_embedding", 0, 0, {}], ["item_tag_embedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_17", "inbound_nodes": [[["embedding_concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dnn_layer1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dnn_layer1", "inbound_nodes": [[["flatten_17", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": null}, "name": "dropout_24", "inbound_nodes": [[["dnn_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dnn_layer2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dnn_layer2", "inbound_nodes": [[["dropout_24", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": null}, "name": "dropout_25", "inbound_nodes": [[["dnn_layer2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_16", "inbound_nodes": [[["embedding_concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dnn_layer3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dnn_layer3", "inbound_nodes": [[["dropout_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fm_linear", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fm_linear", "inbound_nodes": [[["flatten_16", 0, 0, {}]]]}, {"class_name": "crosslayer", "config": {"name": "fm_cross", "trainable": true, "dtype": "float32"}, "name": "fm_cross", "inbound_nodes": [[["embedding_concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": null}, "name": "dropout_26", "inbound_nodes": [[["dnn_layer3", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "fm_combine", "trainable": true, "dtype": "float32"}, "name": "fm_combine", "inbound_nodes": [[["fm_linear", 0, 0, {}], ["fm_cross", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dnn_layer4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dnn_layer4", "inbound_nodes": [[["dropout_26", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_8", "trainable": true, "dtype": "float32"}, "name": "add_8", "inbound_nodes": [[["fm_combine", 0, 0, {}], ["dnn_layer4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "sigmoid", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "sigmoid", "inbound_nodes": [[["add_8", 0, 0, {}]]]}], "input_layers": [["user_id", 0, 0], ["user_type", 0, 0], ["member_type", 0, 0], ["item_id", 0, 0], ["item_catalog", 0, 0], ["item_tag", 0, 0]], "output_layers": [["sigmoid", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 11]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 11]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "user_id"}, "name": "user_id", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "member_type"}, "name": "member_type", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "user_type"}, "name": "user_type", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_id"}, "name": "item_id", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_catalog"}, "name": "item_catalog", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_tag"}, "name": "item_tag", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "user_id_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 60000, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "user_id_embedding", "inbound_nodes": [[["user_id", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "member_type_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 5, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "member_type_embedding", "inbound_nodes": [[["member_type", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "user_type_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "user_type_embedding", "inbound_nodes": [[["user_type", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "item_id_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 300, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "item_id_embedding", "inbound_nodes": [[["item_id", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "item_catalog_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 14, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "item_catalog_embedding", "inbound_nodes": [[["item_catalog", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "item_tag_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 301, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "item_tag_embedding", "inbound_nodes": [[["item_tag", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "embedding_concatenate", "trainable": true, "dtype": "float32", "axis": 1}, "name": "embedding_concatenate", "inbound_nodes": [[["user_id_embedding", 0, 0, {}], ["member_type_embedding", 0, 0, {}], ["user_type_embedding", 0, 0, {}], ["item_id_embedding", 0, 0, {}], ["item_catalog_embedding", 0, 0, {}], ["item_tag_embedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_17", "inbound_nodes": [[["embedding_concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dnn_layer1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dnn_layer1", "inbound_nodes": [[["flatten_17", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": null}, "name": "dropout_24", "inbound_nodes": [[["dnn_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dnn_layer2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dnn_layer2", "inbound_nodes": [[["dropout_24", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": null}, "name": "dropout_25", "inbound_nodes": [[["dnn_layer2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_16", "inbound_nodes": [[["embedding_concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dnn_layer3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dnn_layer3", "inbound_nodes": [[["dropout_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fm_linear", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fm_linear", "inbound_nodes": [[["flatten_16", 0, 0, {}]]]}, {"class_name": "crosslayer", "config": {"name": "fm_cross", "trainable": true, "dtype": "float32"}, "name": "fm_cross", "inbound_nodes": [[["embedding_concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": null}, "name": "dropout_26", "inbound_nodes": [[["dnn_layer3", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "fm_combine", "trainable": true, "dtype": "float32"}, "name": "fm_combine", "inbound_nodes": [[["fm_linear", 0, 0, {}], ["fm_cross", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dnn_layer4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dnn_layer4", "inbound_nodes": [[["dropout_26", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_8", "trainable": true, "dtype": "float32"}, "name": "add_8", "inbound_nodes": [[["fm_combine", 0, 0, {}], ["dnn_layer4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "sigmoid", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "sigmoid", "inbound_nodes": [[["add_8", 0, 0, {}]]]}], "input_layers": [["user_id", 0, 0], ["user_type", 0, 0], ["member_type", 0, 0], ["item_id", 0, 0], ["item_catalog", 0, 0], ["item_tag", 0, 0]], "output_layers": [["sigmoid", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "AUC", "config": {"name": "auc_8", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}, {"class_name": "Recall", "config": {"name": "recall_8", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
х"т
_tf_keras_input_layerТ{"class_name": "InputLayer", "name": "user_id", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "user_id"}}
э"ъ
_tf_keras_input_layerЪ{"class_name": "InputLayer", "name": "member_type", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "member_type"}}
щ"ц
_tf_keras_input_layerЦ{"class_name": "InputLayer", "name": "user_type", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "user_type"}}
х"т
_tf_keras_input_layerТ{"class_name": "InputLayer", "name": "item_id", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_id"}}
я"ь
_tf_keras_input_layerЬ{"class_name": "InputLayer", "name": "item_catalog", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_catalog"}}
щ"ц
_tf_keras_input_layerЦ{"class_name": "InputLayer", "name": "item_tag", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_tag"}}

"
embeddings
#regularization_losses
$	variables
%trainable_variables
&	keras_api
+Ћ&call_and_return_all_conditional_losses
Ќ__call__"љ
_tf_keras_layerп{"class_name": "Embedding", "name": "user_id_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "user_id_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 60000, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}

'
embeddings
(regularization_losses
)	variables
*trainable_variables
+	keras_api
+­&call_and_return_all_conditional_losses
Ў__call__"§
_tf_keras_layerу{"class_name": "Embedding", "name": "member_type_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "member_type_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 5, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}

,
embeddings
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+Џ&call_and_return_all_conditional_losses
А__call__"љ
_tf_keras_layerп{"class_name": "Embedding", "name": "user_type_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "user_type_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}

1
embeddings
2regularization_losses
3	variables
4trainable_variables
5	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"ї
_tf_keras_layerн{"class_name": "Embedding", "name": "item_id_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "item_id_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 300, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
Ё
6
embeddings
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"
_tf_keras_layerц{"class_name": "Embedding", "name": "item_catalog_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "item_catalog_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 14, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}

;
embeddings
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"њ
_tf_keras_layerр{"class_name": "Embedding", "name": "item_tag_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "item_tag_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 301, "output_dim": 16, "embeddings_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11]}}
С
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+З&call_and_return_all_conditional_losses
И__call__"А
_tf_keras_layer{"class_name": "Concatenate", "name": "embedding_concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_concatenate", "trainable": true, "dtype": "float32", "axis": 1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 16]}, {"class_name": "TensorShape", "items": [null, 1, 16]}, {"class_name": "TensorShape", "items": [null, 1, 16]}, {"class_name": "TensorShape", "items": [null, 1, 16]}, {"class_name": "TensorShape", "items": [null, 1, 16]}, {"class_name": "TensorShape", "items": [null, 11, 16]}]}
ъ
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"й
_tf_keras_layerП{"class_name": "Flatten", "name": "flatten_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ф

Hkernel
Ibias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"Н
_tf_keras_layerЃ{"class_name": "Dense", "name": "dnn_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dnn_layer1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
щ
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_24", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": null}}
ф

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"Н
_tf_keras_layerЃ{"class_name": "Dense", "name": "dnn_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dnn_layer2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
щ
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": null}}
ъ
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"й
_tf_keras_layerП{"class_name": "Flatten", "name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ф

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"Н
_tf_keras_layerЃ{"class_name": "Dense", "name": "dnn_layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dnn_layer3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
у

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"М
_tf_keras_layerЂ{"class_name": "Dense", "name": "fm_linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fm_linear", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}

lregularization_losses
m	variables
ntrainable_variables
o	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"ѕ
_tf_keras_layerл{"class_name": "crosslayer", "name": "fm_cross", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fm_cross", "trainable": true, "dtype": "float32"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16]}}
щ
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.9, "noise_shape": null, "seed": null}}
Г
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"Ђ
_tf_keras_layer{"class_name": "Add", "name": "fm_combine", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fm_combine", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
л

xkernel
yregularization_losses
z	variables
{trainable_variables
|	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"О
_tf_keras_layerЄ{"class_name": "Dense", "name": "dnn_layer4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dnn_layer4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 3.200000047683716}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Њ
}regularization_losses
~	variables
trainable_variables
	keras_api
+б&call_and_return_all_conditional_losses
в__call__"
_tf_keras_layerў{"class_name": "Add", "name": "add_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_8", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
д
regularization_losses
	variables
trainable_variables
	keras_api
+г&call_and_return_all_conditional_losses
д__call__"П
_tf_keras_layerЅ{"class_name": "Activation", "name": "sigmoid", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "sigmoid", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}

	iter
beta_1
beta_2

decay
learning_rate"m'm,m1m6m;mHmImRmSm`mamfmgmxm"v'v,v1v6v;vHvIv RvЁSvЂ`vЃavЄfvЅgvІxvЇ"
	optimizer
h
е0
ж1
з2
и3
й4
к5
л6
м7
н8"
trackable_list_wrapper

"0
'1
,2
13
64
;5
H6
I7
R8
S9
`10
a11
f12
g13
x14"
trackable_list_wrapper

"0
'1
,2
13
64
;5
H6
I7
R8
S9
`10
a11
f12
g13
x14"
trackable_list_wrapper
г
metrics
layers
regularization_losses
	variables
layer_metrics
trainable_variables
 layer_regularization_losses
non_trainable_variables
Њ__call__
Ј_default_save_signature
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
-
оserving_default"
signature_map
0:.
рд2user_id_embedding/embeddings
 "
trackable_list_wrapper
'
"0"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
Е
metrics
#regularization_losses
$	variables
 layer_regularization_losses
layer_metrics
%trainable_variables
layers
non_trainable_variables
Ќ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
2:02 member_type_embedding/embeddings
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
'
'0"
trackable_list_wrapper
Е
metrics
(regularization_losses
)	variables
 layer_regularization_losses
layer_metrics
*trainable_variables
layers
non_trainable_variables
Ў__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
0:.2user_type_embedding/embeddings
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
Е
metrics
-regularization_losses
.	variables
 layer_regularization_losses
layer_metrics
/trainable_variables
layers
non_trainable_variables
А__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
/:-	Ќ2item_id_embedding/embeddings
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
'
10"
trackable_list_wrapper
Е
metrics
2regularization_losses
3	variables
 layer_regularization_losses
 layer_metrics
4trainable_variables
Ёlayers
Ђnon_trainable_variables
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
3:12!item_catalog_embedding/embeddings
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
'
60"
trackable_list_wrapper
Е
Ѓmetrics
7regularization_losses
8	variables
 Єlayer_regularization_losses
Ѕlayer_metrics
9trainable_variables
Іlayers
Їnon_trainable_variables
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
0:.	­2item_tag_embedding/embeddings
 "
trackable_list_wrapper
'
;0"
trackable_list_wrapper
'
;0"
trackable_list_wrapper
Е
Јmetrics
<regularization_losses
=	variables
 Љlayer_regularization_losses
Њlayer_metrics
>trainable_variables
Ћlayers
Ќnon_trainable_variables
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
­metrics
@regularization_losses
A	variables
 Ўlayer_regularization_losses
Џlayer_metrics
Btrainable_variables
Аlayers
Бnon_trainable_variables
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Вmetrics
Dregularization_losses
E	variables
 Гlayer_regularization_losses
Дlayer_metrics
Ftrainable_variables
Еlayers
Жnon_trainable_variables
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
%:#
2dnn_layer1/kernel
:2dnn_layer1/bias
0
е0
ж1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
Е
Зmetrics
Jregularization_losses
K	variables
 Иlayer_regularization_losses
Йlayer_metrics
Ltrainable_variables
Кlayers
Лnon_trainable_variables
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Мmetrics
Nregularization_losses
O	variables
 Нlayer_regularization_losses
Оlayer_metrics
Ptrainable_variables
Пlayers
Рnon_trainable_variables
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
%:#
2dnn_layer2/kernel
:2dnn_layer2/bias
0
з0
и1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
Е
Сmetrics
Tregularization_losses
U	variables
 Тlayer_regularization_losses
Уlayer_metrics
Vtrainable_variables
Фlayers
Хnon_trainable_variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Цmetrics
Xregularization_losses
Y	variables
 Чlayer_regularization_losses
Шlayer_metrics
Ztrainable_variables
Щlayers
Ъnon_trainable_variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ыmetrics
\regularization_losses
]	variables
 Ьlayer_regularization_losses
Эlayer_metrics
^trainable_variables
Юlayers
Яnon_trainable_variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
%:#
2dnn_layer3/kernel
:2dnn_layer3/bias
0
й0
к1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
Е
аmetrics
bregularization_losses
c	variables
 бlayer_regularization_losses
вlayer_metrics
dtrainable_variables
гlayers
дnon_trainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
#:!	2fm_linear/kernel
:2fm_linear/bias
0
л0
м1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
Е
еmetrics
hregularization_losses
i	variables
 жlayer_regularization_losses
зlayer_metrics
jtrainable_variables
иlayers
йnon_trainable_variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
кmetrics
lregularization_losses
m	variables
 лlayer_regularization_losses
мlayer_metrics
ntrainable_variables
нlayers
оnon_trainable_variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
пmetrics
pregularization_losses
q	variables
 рlayer_regularization_losses
сlayer_metrics
rtrainable_variables
тlayers
уnon_trainable_variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
фmetrics
tregularization_losses
u	variables
 хlayer_regularization_losses
цlayer_metrics
vtrainable_variables
чlayers
шnon_trainable_variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
$:"	2dnn_layer4/kernel
(
н0"
trackable_list_wrapper
'
x0"
trackable_list_wrapper
'
x0"
trackable_list_wrapper
Е
щmetrics
yregularization_losses
z	variables
 ъlayer_regularization_losses
ыlayer_metrics
{trainable_variables
ьlayers
эnon_trainable_variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
юmetrics
}regularization_losses
~	variables
 яlayer_regularization_losses
№layer_metrics
trainable_variables
ёlayers
ђnon_trainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓmetrics
regularization_losses
	variables
 єlayer_regularization_losses
ѕlayer_metrics
trainable_variables
іlayers
їnon_trainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
8
ј0
љ1
њ2"
trackable_list_wrapper
ю
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
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26"
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
0
е0
ж1"
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
0
з0
и1"
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
0
й0
к1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
л0
м1"
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
(
н0"
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
П

ћtotal

ќcount
§	variables
ў	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Й"
џtrue_positives
true_negatives
false_positives
false_negatives
	variables
	keras_api"Р!
_tf_keras_metricЅ!{"class_name": "AUC", "name": "auc_8", "dtype": "float32", "config": {"name": "auc_8", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
Ѓ

thresholds
true_positives
false_negatives
	variables
	keras_api"Ф
_tf_keras_metricЉ{"class_name": "Recall", "name": "recall_8", "dtype": "float32", "config": {"name": "recall_8", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
0
ћ0
ќ1"
trackable_list_wrapper
.
§	variables"
_generic_user_object
:Ш (2true_positives
:Ш (2true_negatives
 :Ш (2false_positives
 :Ш (2false_negatives
@
џ0
1
2
3"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
5:3
рд2#Adam/user_id_embedding/embeddings/m
7:52'Adam/member_type_embedding/embeddings/m
5:32%Adam/user_type_embedding/embeddings/m
4:2	Ќ2#Adam/item_id_embedding/embeddings/m
8:62(Adam/item_catalog_embedding/embeddings/m
5:3	­2$Adam/item_tag_embedding/embeddings/m
*:(
2Adam/dnn_layer1/kernel/m
#:!2Adam/dnn_layer1/bias/m
*:(
2Adam/dnn_layer2/kernel/m
#:!2Adam/dnn_layer2/bias/m
*:(
2Adam/dnn_layer3/kernel/m
#:!2Adam/dnn_layer3/bias/m
(:&	2Adam/fm_linear/kernel/m
!:2Adam/fm_linear/bias/m
):'	2Adam/dnn_layer4/kernel/m
5:3
рд2#Adam/user_id_embedding/embeddings/v
7:52'Adam/member_type_embedding/embeddings/v
5:32%Adam/user_type_embedding/embeddings/v
4:2	Ќ2#Adam/item_id_embedding/embeddings/v
8:62(Adam/item_catalog_embedding/embeddings/v
5:3	­2$Adam/item_tag_embedding/embeddings/v
*:(
2Adam/dnn_layer1/kernel/v
#:!2Adam/dnn_layer1/bias/v
*:(
2Adam/dnn_layer2/kernel/v
#:!2Adam/dnn_layer2/bias/v
*:(
2Adam/dnn_layer3/kernel/v
#:!2Adam/dnn_layer3/bias/v
(:&	2Adam/fm_linear/kernel/v
!:2Adam/fm_linear/bias/v
):'	2Adam/dnn_layer4/kernel/v
Ђ2
 __inference__wrapped_model_74545њ
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *щЂх
то
!
user_idџџџџџџџџџ
# 
	user_typeџџџџџџџџџ
%"
member_typeџџџџџџџџџ
!
item_idџџџџџџџџџ
&#
item_catalogџџџџџџџџџ
"
item_tagџџџџџџџџџ
к2з
C__inference_model_17_layer_call_and_return_conditional_losses_75237
C__inference_model_17_layer_call_and_return_conditional_losses_75123
C__inference_model_17_layer_call_and_return_conditional_losses_75817
C__inference_model_17_layer_call_and_return_conditional_losses_75963Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
(__inference_model_17_layer_call_fn_75392
(__inference_model_17_layer_call_fn_76043
(__inference_model_17_layer_call_fn_76003
(__inference_model_17_layer_call_fn_75546Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
L__inference_user_id_embedding_layer_call_and_return_conditional_losses_76052Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_user_id_embedding_layer_call_fn_76059Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њ2ї
P__inference_member_type_embedding_layer_call_and_return_conditional_losses_76068Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
п2м
5__inference_member_type_embedding_layer_call_fn_76075Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј2ѕ
N__inference_user_type_embedding_layer_call_and_return_conditional_losses_76084Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
н2к
3__inference_user_type_embedding_layer_call_fn_76091Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_item_id_embedding_layer_call_and_return_conditional_losses_76100Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_item_id_embedding_layer_call_fn_76107Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћ2ј
Q__inference_item_catalog_embedding_layer_call_and_return_conditional_losses_76116Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
р2н
6__inference_item_catalog_embedding_layer_call_fn_76123Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_item_tag_embedding_layer_call_and_return_conditional_losses_76132Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_item_tag_embedding_layer_call_fn_76139Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њ2ї
P__inference_embedding_concatenate_layer_call_and_return_conditional_losses_76150Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
п2м
5__inference_embedding_concatenate_layer_call_fn_76160Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_flatten_17_layer_call_and_return_conditional_losses_76166Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_flatten_17_layer_call_fn_76171Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dnn_layer1_layer_call_and_return_conditional_losses_76206Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dnn_layer1_layer_call_fn_76215Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
E__inference_dropout_24_layer_call_and_return_conditional_losses_76232
E__inference_dropout_24_layer_call_and_return_conditional_losses_76227Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_24_layer_call_fn_76237
*__inference_dropout_24_layer_call_fn_76242Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
E__inference_dnn_layer2_layer_call_and_return_conditional_losses_76277Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dnn_layer2_layer_call_fn_76286Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
E__inference_dropout_25_layer_call_and_return_conditional_losses_76303
E__inference_dropout_25_layer_call_and_return_conditional_losses_76298Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_25_layer_call_fn_76313
*__inference_dropout_25_layer_call_fn_76308Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
E__inference_flatten_16_layer_call_and_return_conditional_losses_76319Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_flatten_16_layer_call_fn_76324Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dnn_layer3_layer_call_and_return_conditional_losses_76359Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dnn_layer3_layer_call_fn_76368Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_fm_linear_layer_call_and_return_conditional_losses_76402Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_fm_linear_layer_call_fn_76411Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_fm_cross_layer_call_and_return_conditional_losses_76426Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_fm_cross_layer_call_fn_76431Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
E__inference_dropout_26_layer_call_and_return_conditional_losses_76443
E__inference_dropout_26_layer_call_and_return_conditional_losses_76448Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_26_layer_call_fn_76453
*__inference_dropout_26_layer_call_fn_76458Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
E__inference_fm_combine_layer_call_and_return_conditional_losses_76464Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_fm_combine_layer_call_fn_76470Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dnn_layer4_layer_call_and_return_conditional_losses_76489Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dnn_layer4_layer_call_fn_76496Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъ2ч
@__inference_add_8_layer_call_and_return_conditional_losses_76502Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_add_8_layer_call_fn_76508Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_sigmoid_layer_call_and_return_conditional_losses_76513Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_sigmoid_layer_call_fn_76518Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
В2Џ
__inference_loss_fn_0_76529
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_1_76540
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_2_76551
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_3_76562
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_4_76573
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_5_76584
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_6_76595
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_7_76606
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference_loss_fn_8_76617
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Bў
#__inference_signature_wrapper_75650item_catalogitem_iditem_tagmember_typeuser_id	user_type"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 с
 __inference__wrapped_model_74545М"',16;HIRS`afgxѕЂё
щЂх
то
!
user_idџџџџџџџџџ
# 
	user_typeџџџџџџџџџ
%"
member_typeџџџџџџџџџ
!
item_idџџџџџџџџџ
&#
item_catalogџџџџџџџџџ
"
item_tagџџџџџџџџџ
Њ "1Њ.
,
sigmoid!
sigmoidџџџџџџџџџШ
@__inference_add_8_layer_call_and_return_conditional_losses_76502ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
%__inference_add_8_layer_call_fn_76508vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "џџџџџџџџџЇ
E__inference_dnn_layer1_layer_call_and_return_conditional_losses_76206^HI0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dnn_layer1_layer_call_fn_76215QHI0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЇ
E__inference_dnn_layer2_layer_call_and_return_conditional_losses_76277^RS0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dnn_layer2_layer_call_fn_76286QRS0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЇ
E__inference_dnn_layer3_layer_call_and_return_conditional_losses_76359^`a0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dnn_layer3_layer_call_fn_76368Q`a0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЅ
E__inference_dnn_layer4_layer_call_and_return_conditional_losses_76489\x0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dnn_layer4_layer_call_fn_76496Ox0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЇ
E__inference_dropout_24_layer_call_and_return_conditional_losses_76227^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 Ї
E__inference_dropout_24_layer_call_and_return_conditional_losses_76232^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dropout_24_layer_call_fn_76237Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
*__inference_dropout_24_layer_call_fn_76242Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџЇ
E__inference_dropout_25_layer_call_and_return_conditional_losses_76298^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 Ї
E__inference_dropout_25_layer_call_and_return_conditional_losses_76303^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dropout_25_layer_call_fn_76308Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
*__inference_dropout_25_layer_call_fn_76313Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџЇ
E__inference_dropout_26_layer_call_and_return_conditional_losses_76443^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 Ї
E__inference_dropout_26_layer_call_and_return_conditional_losses_76448^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dropout_26_layer_call_fn_76453Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
*__inference_dropout_26_layer_call_fn_76458Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
P__inference_embedding_concatenate_layer_call_and_return_conditional_losses_76150ЕЂ
ћЂї
є№
&#
inputs/0џџџџџџџџџ
&#
inputs/1џџџџџџџџџ
&#
inputs/2џџџџџџџџџ
&#
inputs/3џџџџџџџџџ
&#
inputs/4џџџџџџџџџ
&#
inputs/5џџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 т
5__inference_embedding_concatenate_layer_call_fn_76160ЈЂ
ћЂї
є№
&#
inputs/0џџџџџџџџџ
&#
inputs/1џџџџџџџџџ
&#
inputs/2џџџџџџџџџ
&#
inputs/3џџџџџџџџџ
&#
inputs/4џџџџџџџџџ
&#
inputs/5џџџџџџџџџ
Њ "џџџџџџџџџІ
E__inference_flatten_16_layer_call_and_return_conditional_losses_76319]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 ~
*__inference_flatten_16_layer_call_fn_76324P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
E__inference_flatten_17_layer_call_and_return_conditional_losses_76166]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 ~
*__inference_flatten_17_layer_call_fn_76171P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЭ
E__inference_fm_combine_layer_call_and_return_conditional_losses_76464ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Є
*__inference_fm_combine_layer_call_fn_76470vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "џџџџџџџџџЃ
C__inference_fm_cross_layer_call_and_return_conditional_losses_76426\3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_fm_cross_layer_call_fn_76431O3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЅ
D__inference_fm_linear_layer_call_and_return_conditional_losses_76402]fg0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
)__inference_fm_linear_layer_call_fn_76411Pfg0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџД
Q__inference_item_catalog_embedding_layer_call_and_return_conditional_losses_76116_6/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
6__inference_item_catalog_embedding_layer_call_fn_76123R6/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЏ
L__inference_item_id_embedding_layer_call_and_return_conditional_losses_76100_1/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
1__inference_item_id_embedding_layer_call_fn_76107R1/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџА
M__inference_item_tag_embedding_layer_call_and_return_conditional_losses_76132_;/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
2__inference_item_tag_embedding_layer_call_fn_76139R;/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ:
__inference_loss_fn_0_76529HЂ

Ђ 
Њ " :
__inference_loss_fn_1_76540IЂ

Ђ 
Њ " :
__inference_loss_fn_2_76551RЂ

Ђ 
Њ " :
__inference_loss_fn_3_76562SЂ

Ђ 
Њ " :
__inference_loss_fn_4_76573`Ђ

Ђ 
Њ " :
__inference_loss_fn_5_76584aЂ

Ђ 
Њ " :
__inference_loss_fn_6_76595fЂ

Ђ 
Њ " :
__inference_loss_fn_7_76606gЂ

Ђ 
Њ " :
__inference_loss_fn_8_76617xЂ

Ђ 
Њ " Г
P__inference_member_type_embedding_layer_call_and_return_conditional_losses_76068_'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
5__inference_member_type_embedding_layer_call_fn_76075R'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
C__inference_model_17_layer_call_and_return_conditional_losses_75123И"',16;HIRS`afgx§Ђљ
ёЂэ
то
!
user_idџџџџџџџџџ
# 
	user_typeџџџџџџџџџ
%"
member_typeџџџџџџџџџ
!
item_idџџџџџџџџџ
&#
item_catalogџџџџџџџџџ
"
item_tagџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
C__inference_model_17_layer_call_and_return_conditional_losses_75237И"',16;HIRS`afgx§Ђљ
ёЂэ
то
!
user_idџџџџџџџџџ
# 
	user_typeџџџџџџџџџ
%"
member_typeџџџџџџџџџ
!
item_idџџџџџџџџџ
&#
item_catalogџџџџџџџџџ
"
item_tagџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 њ
C__inference_model_17_layer_call_and_return_conditional_losses_75817В"',16;HIRS`afgxїЂѓ
ыЂч
ми
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
"
inputs/3џџџџџџџџџ
"
inputs/4џџџџџџџџџ
"
inputs/5џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 њ
C__inference_model_17_layer_call_and_return_conditional_losses_75963В"',16;HIRS`afgxїЂѓ
ыЂч
ми
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
"
inputs/3џџџџџџџџџ
"
inputs/4џџџџџџџџџ
"
inputs/5џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 и
(__inference_model_17_layer_call_fn_75392Ћ"',16;HIRS`afgx§Ђљ
ёЂэ
то
!
user_idџџџџџџџџџ
# 
	user_typeџџџџџџџџџ
%"
member_typeџџџџџџџџџ
!
item_idџџџџџџџџџ
&#
item_catalogџџџџџџџџџ
"
item_tagџџџџџџџџџ
p

 
Њ "џџџџџџџџџи
(__inference_model_17_layer_call_fn_75546Ћ"',16;HIRS`afgx§Ђљ
ёЂэ
то
!
user_idџџџџџџџџџ
# 
	user_typeџџџџџџџџџ
%"
member_typeџџџџџџџџџ
!
item_idџџџџџџџџџ
&#
item_catalogџџџџџџџџџ
"
item_tagџџџџџџџџџ
p 

 
Њ "џџџџџџџџџв
(__inference_model_17_layer_call_fn_76003Ѕ"',16;HIRS`afgxїЂѓ
ыЂч
ми
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
"
inputs/3џџџџџџџџџ
"
inputs/4џџџџџџџџџ
"
inputs/5џџџџџџџџџ
p

 
Њ "џџџџџџџџџв
(__inference_model_17_layer_call_fn_76043Ѕ"',16;HIRS`afgxїЂѓ
ыЂч
ми
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
"
inputs/3џџџџџџџџџ
"
inputs/4џџџџџџџџџ
"
inputs/5џџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
B__inference_sigmoid_layer_call_and_return_conditional_losses_76513X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 v
'__inference_sigmoid_layer_call_fn_76518K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЋ
#__inference_signature_wrapper_75650"',16;HIRS`afgxМЂИ
Ђ 
АЊЌ
6
item_catalog&#
item_catalogџџџџџџџџџ
,
item_id!
item_idџџџџџџџџџ
.
item_tag"
item_tagџџџџџџџџџ
4
member_type%"
member_typeџџџџџџџџџ
,
user_id!
user_idџџџџџџџџџ
0
	user_type# 
	user_typeџџџџџџџџџ"1Њ.
,
sigmoid!
sigmoidџџџџџџџџџЏ
L__inference_user_id_embedding_layer_call_and_return_conditional_losses_76052_"/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
1__inference_user_id_embedding_layer_call_fn_76059R"/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџБ
N__inference_user_type_embedding_layer_call_and_return_conditional_losses_76084_,/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
3__inference_user_type_embedding_layer_call_fn_76091R,/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ