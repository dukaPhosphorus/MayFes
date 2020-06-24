from keras.engine import Input
from keras.engine import Model

from keras.layers import Dense, TimeDistributed, LSTM, Dropout, Activation, Lambda, concatenate, add


def deepBach(num_features_lr, num_features_c, num_pitches, num_features_meta, num_units_lstm=[200],
             num_dense=200, timesteps=16):
    """

    :param num_features_lr: size of left or right features vectors
    :param num_features_c: size of central features vectors
    :param num_pitches: size of output
    :param num_units_lstm: list of lstm layer sizes
    :param num_dense:
    :return:
    """
    print('in models_zoo.py deepBach')
    print('*'*100)
    print('num_features_lr=', num_features_lr)
    print('num_features_c=', num_features_c)
    print('num_features_meta=', num_features_meta)
    print('num_pitches=', num_pitches)
    print('timesteps=', timesteps)
    # input
    left_features = Input(shape=(timesteps, num_features_lr), name='left_features')
    right_features = Input(shape=(timesteps, num_features_lr), name='right_features')
    central_features = Input(shape=(num_features_c,), name='central_features')
    # input metadatas
    left_metas = Input(shape=(timesteps, num_features_meta), name='left_metas')
    right_metas = Input(shape=(timesteps, num_features_meta), name='right_metas')
    central_metas = Input(shape=(num_features_meta,), name='central_metas')

    # embedding layer for left and right
    embedding_left = Dense(input_dim=num_features_lr + num_features_meta,
                           output_dim=num_dense, name='embedding_left')
    embedding_right = Dense(input_dim=num_features_lr + num_features_meta,
                            output_dim=num_dense, name='embedding_right')


    
    #print(concatenate([left_features, left_metas]).shape)
    predictions_left = TimeDistributed(embedding_left)(concatenate([left_features, left_metas]))
    #print(predictions_left.shape)      #=(？,16=timesteps, 200=num_dense)  
    predictions_right = TimeDistributed(embedding_right)(concatenate([right_features, right_metas]))

    predictions_center = concatenate([central_features, central_metas])
    #print(predictions_center.shape)  #=(？,95)、※ 95 = sum(num_pitches)
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)
    #print(predictions_center.shape)  #=(?, 200)
    
    return_sequences = True
    for k, stack_index in enumerate(range(len(num_units_lstm))):
        if k == len(num_units_lstm) - 1:
            return_sequences = False
        predictions_left = LSTM(num_units_lstm[stack_index],
                                return_sequences=return_sequences,
                                name='lstm_left_' + str(stack_index)
                                )(predictions_left)
        predictions_right = LSTM(num_units_lstm[stack_index],
                                 return_sequences=return_sequences,
                                 name='lstm_right_' + str(stack_index)
                                 )(predictions_right)

    #print('afterLSTM \n', predictions_left.shape)
    predictions = concatenate([predictions_left, predictions_center, predictions_right])
    #print(predictions.shape)
    predictions = Dense(num_dense, activation='relu')(predictions)
    #print(predictions.shape)
    pitch_prediction = Dense(num_pitches, activation='softmax',
                             name='pitch_prediction')(predictions)

    #print(pitch_prediction.shape)
    model = Model(input=[left_features, central_features, right_features,
                         left_metas, central_metas, right_metas
                         ],
                  output=pitch_prediction)

    model.compile(optimizer='adam',
                  loss={'pitch_prediction': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    #model.summary()
    return model




















def deepBach_chord(num_features_lr, num_features_c, num_pitches, num_features_meta, num_localseqs_lr, num_units_lstm=[200],
             num_dense=200, timesteps=16):
    """

    :param num_features_lr: size of left or right features vectors
    :param num_features_c: size of central features vectors
    :param num_pitches: size of output
    :param num_units_lstm: list of lstm layer sizes
    :param num_dense:
    :return:
    """
    print('in models_zoo.py deepBach')
    print('*'*100)
    print('num_features_lr=', num_features_lr)
    print('num_features_c=', num_features_c)
    print('num_features_meta=', num_features_meta)
    print('num_pitches=', num_pitches)
    print('timesteps=', timesteps)
    print('num_localseqs_lr=', num_localseqs_lr)
    # input
    left_features = Input(shape=(timesteps, num_features_lr), name='left_features')
    right_features = Input(shape=(timesteps, num_features_lr), name='right_features')
    central_features = Input(shape=(num_features_c,), name='central_features')

    # input metadatas
    left_metas = Input(shape=(timesteps, num_features_meta), name='left_metas')
    right_metas = Input(shape=(timesteps, num_features_meta), name='right_metas')
    central_metas = Input(shape=(num_features_meta,), name='central_metas')


    
    ######################### input local datas ##########################
    left_local_seqs = Input(shape=(4, num_localseqs_lr), name='left_local_seqs')       ##5=local_seq_length 
    right_local_seqs = Input(shape=(4, num_localseqs_lr), name='right_local_seqs')     ##5=local_seq_length 
    ######################################################################


    
    # embedding layer for left and right
    embedding_left = Dense(input_dim=num_features_lr + num_features_meta,
                           output_dim=num_dense, name='embedding_left')
    embedding_right = Dense(input_dim=num_features_lr + num_features_meta,
                            output_dim=num_dense, name='embedding_right')


    
    ############## embedding layer for local left and right ###############
    embedding_local_left = Dense(input_dim=num_localseqs_lr,
                           output_dim=num_dense, name='embedding_local_left')
    embedding_local_right = Dense(input_dim=num_localseqs_lr,
                            output_dim=num_dense, name='embedding_local_right')
    #######################################################################


    
    
    #print(concatenate([left_features, left_metas]).shape)
    predictions_left = TimeDistributed(embedding_left)(concatenate([left_features, left_metas]))
    #print(predictions_left.shape)      #=(？,16=timesteps, 200=num_dense)  
    predictions_right = TimeDistributed(embedding_right)(concatenate([right_features, right_metas]))

    predictions_center = concatenate([central_features, central_metas])
    #print(predictions_center.shape)  #=(？,95)、※ 95 = sum(num_pitches)
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)
    #print(predictions_center.shape)  #=(?, 200)
    ####################################################これ一つ層減らしてもよいのでは


    
    ################## predictions local left and right ############################
    predictions_local_left = TimeDistributed(embedding_local_left)(left_local_seqs)
    predictions_local_right = TimeDistributed(embedding_local_right)(right_local_seqs)
    ################################################################################




    
    return_sequences = True
    for k, stack_index in enumerate(range(len(num_units_lstm))):
        if k == len(num_units_lstm) - 1:
            return_sequences = False
        predictions_left = LSTM(num_units_lstm[stack_index],
                                return_sequences=return_sequences,
                                name='lstm_left_' + str(stack_index)
                                )(predictions_left)
        predictions_right = LSTM(num_units_lstm[stack_index],
                                 return_sequences=return_sequences,
                                 name='lstm_right_' + str(stack_index)
                                 )(predictions_right)
        
    ################## LSTMs for local left and right ##############################
    #predictions_local_left = LSTM(80,
    #                            return_sequences=True,
    #                              name='lstm_local_left_0')(predictions_local_left)
    #predictions_local_right = LSTM(80,
    #                            return_sequences=True,
    #                              name='lstm_local_right_0')(predictions_local_right)

    predictions_local_left = LSTM(80,
                                return_sequences=False,
                                  name='lstm_local_left_1')(predictions_local_left)
    predictions_local_right = LSTM(80,
                                return_sequences=False,
                                  name='lstm_local_right_1')(predictions_local_right)

    ################################################################################

    

    ########################## ドロップアウト #############################
    predictions_left = Dropout(0.3)(predictions_left)
    predictions_right = Dropout(0.3)(predictions_right)
    predictions_local_left = Dropout(0.6)(predictions_local_left)
    predictions_local_right = Dropout(0.6)(predictions_local_right)
    ####################################################################
    
    #print('afterLSTM \n', predictions_left.shape)
    predictions = concatenate([predictions_left, predictions_center, predictions_right])
    #print(predictions.shape)
    predictions = Dense(num_dense, activation='relu')(predictions)
    #print(predictions.shape)


    
    ################### concatenate local left and right ################################
    local_predictions = concatenate([predictions_local_left,predictions_local_right])
    local_predictions = Dense(num_dense, activation='relu')(local_predictions)
    #####################################################################################


    
    #################  concatenate local and main predictions #######################
    total_predictions = concatenate([predictions, local_predictions])
    total_predictions = Dense(num_dense, activation='relu')(total_predictions)
    ################################################################################

    
    pitch_prediction = Dense(num_pitches, activation='softmax',
                             name='pitch_prediction')(total_predictions)


    

    #print(pitch_prediction.shape)
    model = Model(input=[left_features, central_features, right_features,
                         left_metas, central_metas, right_metas, left_local_seqs, right_local_seqs
                         ],
                  output=pitch_prediction)

    model.compile(optimizer='adam',
                  loss={'pitch_prediction': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    #model.summary()
    return model

























































def deepbach_skip_connections(num_features_lr, num_features_c, num_features_meta, num_pitches, num_units_lstm=[200],
                              num_dense=200, timesteps=16):
    """

    :param num_features_lr: size of left or right features vectors
    :param num_features_c: size of central features vectors
    :param num_pitches: size of output
    :param num_units_lstm: list of lstm layer sizes
    :param num_dense:
    :return:
    """
    left_features = Input(shape=(timesteps, num_features_lr), name='left_features')
    right_features = Input(shape=(timesteps, num_features_lr), name='right_features')
    central_features = Input(shape=(num_features_c,), name='central_features')

    left_metas = Input(shape=(timesteps, num_features_meta), name='left_metas')
    right_metas = Input(shape=(timesteps, num_features_meta), name='right_metas')
    central_metas = Input(shape=(num_features_meta,), name='central_metas')

    # embedding layer for left and right
    embedding_left = Dense(input_dim=num_features_lr + num_features_meta,
                           units=num_dense, name='embedding_left')
    embedding_right = Dense(input_dim=num_features_lr + num_features_meta,
                            units=num_dense, name='embedding_right')

    # merge features and metadata

    predictions_left = concatenate([left_features, left_metas])
    predictions_right = concatenate([right_features, right_metas])
    predictions_center = concatenate([central_features, central_metas])

    # input dropout
    predictions_left = Dropout(0.2)(predictions_left)
    predictions_right = Dropout(0.2)(predictions_right)
    predictions_center = Dropout(0.2)(predictions_center)

    # embedding
    predictions_left = TimeDistributed(embedding_left)(predictions_left)
    predictions_right = TimeDistributed(embedding_right)(predictions_right)

    # central NN
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)
    # predictions_center = Dropout(0.5)(predictions_center)
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)

    # left and right recurrent networks
    return_sequences = True
    for k, stack_index in enumerate(range(len(num_units_lstm))):
        if k == len(num_units_lstm) - 1:
            return_sequences = False

        if k > 0:
            # todo difference between concat and sum
            predictions_left_tmp = add([Activation('relu')(predictions_left), predictions_left_old])
            predictions_right_tmp = add([Activation('relu')(predictions_right), predictions_right_old])
        else:
            predictions_left_tmp = predictions_left
            predictions_right_tmp = predictions_right

        predictions_left_old = predictions_left
        predictions_right_old = predictions_right
        predictions_left = predictions_left_tmp
        predictions_right = predictions_right_tmp

        predictions_left = LSTM(num_units_lstm[stack_index],
                                return_sequences=return_sequences,
                                name='lstm_left_' + str(stack_index)
                                )(predictions_left)

        predictions_right = LSTM(num_units_lstm[stack_index],
                                 return_sequences=return_sequences,
                                 name='lstm_right_' + str(stack_index)
                                 )(predictions_right)

        # todo dropout here?
        # predictions_left = Dropout(0.5)(predictions_left)
        # predictions_right = Dropout(0.5)(predictions_right)

    # retain only last input for skip connections
    predictions_left_old = Lambda(lambda t: t[:, -1, :],
                                  output_shape=lambda input_shape: (input_shape[0], input_shape[-1])
                                  )(predictions_left_old)
    predictions_right_old = Lambda(lambda t: t[:, -1, :],
                                   output_shape=lambda input_shape: (input_shape[0], input_shape[-1],)
                                   )(predictions_right_old)
    # concat or sum
    predictions_left = concatenate([Activation('relu')(predictions_left), predictions_left_old])
    predictions_right = concatenate([Activation('relu')(predictions_right), predictions_right_old])

    predictions = concatenate([predictions_left, predictions_center, predictions_right])
    predictions = Dense(num_dense, activation='relu')(predictions)
    pitch_prediction = Dense(num_pitches, activation='softmax',
                             name='pitch_prediction')(predictions)

    model = Model(inputs=[left_features, central_features, right_features,

                         left_metas, right_metas, central_metas],
                  outputs=pitch_prediction)

    model.compile(optimizer='adam',
                  loss={'pitch_prediction': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    model.summary()
    return model
