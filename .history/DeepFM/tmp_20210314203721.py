def Item2vec(X,hidden_units=[128,128,128,128]):
    #string_col=['item_name','tag','interestb4pr','flexible_return','tax_rating']
    #int_col=['term','age_lower','age_upper','holder_identity','establish_yr','fapiao']
    #cont_col=['credit','rate_lower','rate_upper','fapiao_income']
    #string col
    item_name_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    output_sequence_length=1,name='item_name_vectorize',vocabulary=np.unique(X['item_id']))
    item_name_embedding=tf.keras.layers.Embedding(len(np.unique(X['item_id']))+2,embedding_dim,name='item_name_embedding')
    tag_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    output_sequence_length=2,name='tag_vectorize')
    tag_vectorize.adapt(X['tag'])
    tag_embedding=tf.keras.layers.Embedding(tag_tokens,embedding_dim,name='tag_embedding')

    interestb4pr_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    name='interestb4pr_vectorize',vocabulary=np.unique(X['interestb4pr']),output_sequence_length=1)
    interestb4pr_embedding=tf.keras.layers.Embedding(len(np.unique(X['interestb4pr']))+2,embedding_dim,name='interestb4pr_embedding')
    
    flexible_return_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    name='flexible_return_vectorize',vocabulary=np.unique(X['flexible_return']),output_sequence_length=1)
    flexible_return_embedding=tf.keras.layers.Embedding(len(np.unique(X['flexible_return']))+2,embedding_dim,name='flexible_return_embedding')

    tax_rating_vectorize=tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int',
    output_sequence_length=6,name='tax_rating_vectorize')
    tax_rating_embedding=tf.keras.layers.Embedding(len(np.unique(X['tax_rating']))+2,embedding_dim,name='tax_rating_embedding')
    tax_rating_vectorize.adapt(X['tax_rating'])
    
    #integer col
    term_vectorize=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['term']),name='term_vectorize',mask_value=None)
    term_embedding=tf.keras.layers.Embedding(len(np.unique(X['term']))+2,embedding_dim,name='term_embedding')

    age_lower_vectorize=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['age_lower']),name='age_lower_vectorize',mask_value=None)
    age_lower_embedding=tf.keras.layers.Embedding(len(np.unique(X['age_lower']))+2,embedding_dim,name='age_lower_embedding')

    age_upper_vectorize=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['age_upper']),name='age_upper_vectorize',mask_value=None)
    age_upper_embedding=tf.keras.layers.Embedding(len(np.unique(X['age_upper']))+2,embedding_dim,name='age_upper_embedding')

    holder_identity_vectorize=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['holder_identity']),name='holder_identity_vectorize',mask_value=None)
    holder_identity_embedding=tf.keras.layers.Embedding(len(np.unique(X['holder_identity']))+2,embedding_dim,name='holder_identity_embedding')

    establish_yr_vectorize=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['establish_yr']),name='establish_yr_vectorize',mask_value=None)
    establish_yr_embedding=tf.keras.layers.Embedding(len(np.unique(X['establish_yr']))+2,embedding_dim,name='establish_yr_embedding')

    fapiao_vectorize=tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=np.unique(X['fapiao']),name='fapiao_vectorize',mask_value=None)
    fapiao_embedding=tf.keras.layers.Embedding(len(np.unique(X['fapiao']))+2,embedding_dim,name='fapiao_embedding')
    
    #string input
    item_name=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='item_id')
    tag=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='tag')
    interestb4pr=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='interestb4pr')
    flexible_return=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='flexible_return')
    tax_rating=tf.keras.layers.Input(shape=(1,),dtype=tf.string,name='tax_rating')
    
    #integer input
    term=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='term')
    age_lower=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='age_lower')
    age_upper=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='age_upper')
    holder_identity=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='holder_identity')
    establish_yr=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='establish_yr')
    fapiao=tf.keras.layers.Input(shape=(1,),dtype=tf.int64,name='fapiao')
    #continuous input
    credit=tf.keras.layers.Input(shape=(1,),dtype=tf.float32,name='credit')
    rate_lower=tf.keras.layers.Input(shape=(1,),dtype=tf.float32,name='rate_lower')
    rate_upper=tf.keras.layers.Input(shape=(1,),dtype=tf.float32,name='rate_upper')
    fapiao_income=tf.keras.layers.Input(shape=(1,),dtype=tf.float32,name='fapiao_income')
    #embeddings
    #string embedding
    #item
    item_name_embedding=item_name_embedding(item_name_vectorize(item_name))
    tag_embedding=tag_embedding(tag_vectorize(tag))
    interestb4pr_embedding=interestb4pr_embedding(interestb4pr_vectorize(interestb4pr))
    flexible_return_embedding=flexible_return_embedding(flexible_return_vectorize(flexible_return))
    tax_rating_embedding=tax_rating_embedding(tax_rating_vectorize(tax_rating))
    #integer embedding
    term_embedding=term_embedding(term_vectorize(term))
    age_lower_embedding=age_lower_embedding(age_lower_vectorize(age_lower))
    age_upper_embedding=age_upper_embedding(age_upper_vectorize(age_upper))
    holder_identity_embedding=holder_identity_embedding(holder_identity_vectorize(holder_identity))
    establish_yr_embedding=establish_yr_embedding(establish_yr_vectorize(establish_yr))
    fapiao_embedding=fapiao_embedding(fapiao_vectorize(fapiao))

    item_vector=tf.keras.layers.GlobalAveragePooling1D()(tf.concat([item_name_embedding,tag_embedding,interestb4pr_embedding,flexible_return_embedding,
    flexible_return_embedding,tax_rating_embedding,term_embedding,age_lower_embedding,age_upper_embedding,holder_identity_embedding,establish_yr_embedding,fapiao_embedding],axis=1))

    vector=tf.keras.layers.concatenate([item_vector,credit,rate_lower,rate_upper,fapiao_income],axis=1)

    DNN=tf.keras.Sequential(name='DNN')
    for layer_size in hidden_units:
        DNN.add(tf.keras.layers.Dense(layer_size,activation='relu'))
    DNN.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    pred=DNN(vector)

    model=tf.keras.Model(inputs=[item_name,tag,interestb4pr,flexible_return,tax_rating,term,age_lower,age_upper,holder_identity,
                                establish_yr,fapiao,credit,rate_lower,rate_upper,fapiao_income],outputs=pred)
    return model