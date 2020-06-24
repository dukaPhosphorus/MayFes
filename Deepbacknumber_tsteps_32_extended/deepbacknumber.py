"""
Created on 15 mars 2016

@author: Gaetan Hadjeres
"""
import argparse
import os
import pickle
import music21

from keras.models import model_from_json, model_from_yaml
from models_zoo import deepBach, deepbach_skip_connections, deepBach_chord
from music21 import midi, converter, metadata
from tqdm import tqdm

from data_utils import initialization, START_SYMBOL, END_SYMBOL, all_metadatas, standard_note, SOP, BASS, BACKNUMBER_DATASET, generator_from_raw_dataset, all_features, indexed_chorale_to_score, melody_to_inputs, make_local_sequences#, pickled_dataset_path, \,
    
from metadata import *


def generation(model_base_name, models, timesteps, melody=None, chorale_metas=None,
               initial_seq=None, temperature=1.0, parallel=False, batch_size_per_voice=8, num_iterations=None,
               sequence_length=160,
               output_file=None, pickled_dataset=BACKNUMBER_DATASET):

    _, _, _, _, note2indexes, _, _ = pickle.load(open(pickled_dataset, 'rb'))

    
    # Test by generating a sequence

    # todo -p parameter
    parallel = True
    if parallel:
        seq = parallel_gibbs(models=models, model_base_name=model_base_name,
                             melody=melody, chorale_metas=chorale_metas, timesteps=timesteps,
                             num_iterations=num_iterations, sequence_length=sequence_length,
                             temperature=temperature,
                             initial_seq=initial_seq, batch_size_per_voice=batch_size_per_voice,
                             parallel_updates=True, pickled_dataset=pickled_dataset)
    else:
        # todo refactor
        print('gibbs function must be refactored!')
        # seq = gibbs(models=models, model_base_name=model_base_name,
        #             timesteps=timesteps,
        #             melody=melody, fermatas_melody=fermatas_melody,
        #             num_iterations=num_iterations, sequence_length=sequence_length,
        #             temperature=temperature,
        #             initial_seq=initial_seq,
        #             pickled_dataset=pickled_dataset)
        raise NotImplementedError

    # convert
    print(seq)
    chord_slur_index = note2indexes[1]['__']
    bar_num = len(seq)//4//SUBDIVISION
    
    '''
    for bar_index in range (bar_num):
        bar = seq[4*SUBDIVISION*bar_index:4*SUBDIVISION*(bar_index+1)]
        if bar[15][1] == bar[11][1]:seq[4*SUBDIVISION*bar_index+15][1] = chord_slur_index
        if bar[11][1] == bar[7][1]: seq[4*SUBDIVISION*bar_index+11][1] = chord_slur_index
        if bar[7][1] == bar[3][1]:  seq[4*SUBDIVISION*bar_index+7][1] = chord_slur_index
        if bar[3][1] == bar[0][1]:  seq[4*SUBDIVISION*bar_index+3][1] = chord_slur_index
        
        time = 0
        if chord_slur_index > 0:
            chord_index =0
        else:
            chord_index = 1
        while (time<16):
            if ((chord_index != bar[time][1]) and (bar[time][1]!=chord_slur_index)):
                chord_index = bar[time][1]
            elif ((chord_index == bar[time][1]) and (bar[time][1]!=chord_slur_index)):
                seq[4*SUBDIVISION*bar_index+time][1]= chord_slur_index
            time = time + 1
        '''
        
        
        
        
    #print('seq.shape=', seq.shape)
    #print(np.transpose(seq, axes=(1, 0)))
    
    print('making_score_from_generated_seq')
    score = indexed_chorale_to_score(np.transpose(seq, axes=(1, 0)),
                                     pickled_dataset=pickled_dataset
                                     )

    # save as MIDI file
    if output_file:
        mf = midi.translate.music21ObjectToMidiFile(score)
        mf.open(output_file, 'wb')
        mf.write()
        mf.close()
        print("File " + output_file + " written")

    # display in editor
    print('score_showing')

    if melody is not None:
        score = score.transpose(-2)

    md = metadata.Metadata()
    md.title = 'back number風の曲が生成されました'
    md.composer = ""#'生田絵梨花'
    score.metadata = md
    score.show()
    return seq




#generationの中では以下の形で parallel_gibbs　を読み込み
#model_base_name = deepbach_custom みたいな
"""
seq = parallel_gibbs(models=models, model_base_name=model_base_name,
                             melody=melody, chorale_metas=chorale_metas, timesteps=timesteps,
                             num_iterations=num_iterations, sequence_length=sequence_length,
                             temperature=temperature,
                             initial_seq=initial_seq, batch_size_per_voice=batch_size_per_voice,
                             parallel_updates=True, pickled_dataset=pickled_dataset)

"""
def parallel_gibbs(models=None, melody=None, chorale_metas=None, sequence_length=50, num_iterations=1000,
                   timesteps=16,
                   model_base_name='models/raw_dataset/tmp/',
                   temperature=1., initial_seq=None, batch_size_per_voice=16, parallel_updates=True,
                   pickled_dataset=BACKNUMBER_DATASET):
    """
    samples from models in model_base_name
    """

    X, X_metadatas, voices_ids, index2notes, note2indexes, metadatas,chordname2chordset = pickle.load(open(pickled_dataset, 'rb'))
    num_pitches = list(map(len, index2notes))
    num_voices = len(voices_ids)
    # load models if not
    if models is None:
        for expert_index in range(num_voices):
            model_name = model_base_name + str(expert_index)

            model = load_model(model_name=model_name, yaml=False)
            models.append(model)

    # initialization sequence
    if melody is not None:
        sequence_length = len(melody)
        if chorale_metas is not None:
            sequence_length = min(sequence_length, len(chorale_metas[0]))
    elif chorale_metas is not None:
        sequence_length = len(chorale_metas[0])

    #print(chorale_metas)
    #print(sequence_length)
    
    seq = np.zeros(shape=(2 * timesteps + sequence_length, num_voices))
    #print(seq)
    for expert_index in range(num_voices):
        # Add start and end symbol + random init
        seq[:timesteps, expert_index] = [note2indexes[expert_index][START_SYMBOL]] * timesteps
        #はじめのtimesteps分だけSTART_SYMBOLに対応する数字で埋める
        seq[timesteps:-timesteps, expert_index] = np.random.randint(num_pitches[expert_index],
                                                                    size=sequence_length)
        if expert_index == 1:######################################################################追加さすがにコードの初期値はこうして良いのでは？4分音符刻みのところ以外ははじめからスラーシンボルで埋めとく
            insert_seq = np.full(sequence_length, note2indexes[1]['__'])
            for i in range (len(insert_seq)):
                if i%4 == 0:
                    insert_seq[i] = np.random.randint(num_pitches[expert_index])
            seq[timesteps:-timesteps, expert_index] = insert_seq
        ######################################################################ここまで追加
        #間(楽譜に反映される部分)はとりあえず適当な数字で埋める
        seq[-timesteps:, expert_index] = [note2indexes[expert_index][END_SYMBOL]] * timesteps
        #終わりのtimesteps分だけEND_SYMBOLに対応する数字で埋める

    #print(seq) #この時点では初期値
    
    if initial_seq is not None:
        seq = initial_seq
        min_voice = 1
        # works only with reharmonization
    
    if melody is not None:
        seq[timesteps:-timesteps, 0] = melody
        min_voice = 1
    else:
        min_voice = 0

        
    if chorale_metas is not None:
        # chorale_metas is a list
        extended_chorale_metas = [np.concatenate((np.zeros((timesteps,)),
                                                  chorale_meta,
                                                  np.zeros((timesteps,))),
                                                 axis=0)
                                  for chorale_meta in chorale_metas]
    else:
        raise NotImplementedError
    #extended_chorale_metasはchoralemetasを0拡張したもの
    
    min_temperature = temperature
    temperature = 1.5

    # Main loop   #num_iterationsは自分で -i で入力した値を生成する曲のパート数の数で割ったもの
    for iteration in tqdm(range(num_iterations)):
        temperature = max(min_temperature, temperature * 0.9992)  # Recuit
        #print(temperature)

        time_indexes = {}
        probas = {}
        for voice_index in range(min_voice, num_voices):
            batch_input_features = []
            time_indexes[voice_index] = []

            for batch_index in range(batch_size_per_voice):
                time_index = np.random.randint(timesteps, sequence_length + timesteps)
                time_indexes[voice_index].append(time_index)
                #print(time_index)
                

                (left_feature,
                 central_feature,
                 right_feature,
                 label) = all_features(seq, voice_index, time_index, timesteps, num_pitches, num_voices)

                left_local_seq, right_local_seq = make_local_sequences(seq, voice_index, time_index, num_pitches, note2indexes)
                
                left_metas, central_metas, right_metas = all_metadatas(chorale_metadatas=extended_chorale_metas,
                                                                       metadatas=metadatas,
                                                                       time_index=time_index, timesteps=timesteps)

                input_features = {'left_features': left_feature[:, :],
                                  'central_features': central_feature[:],
                                  'right_features': right_feature[:, :],
                                  'left_metas': left_metas,
                                  'central_metas': central_metas,
                                  'right_metas': right_metas,
                                  'left_local_seqs':left_local_seq[:, :],
                                  'right_local_seqs':right_local_seq[:, :]}
                #print('='*10, voice_index, '='*10)
                #print(input_features)

                # list of dicts: predict need dict of numpy arrays
                batch_input_features.append(input_features)
            
            # convert input_features
            batch_input_features = {key: np.array([input_features[key] for input_features in batch_input_features])
                                    for key in batch_input_features[0].keys()
                                    }
            #print(batch_input_features)
            
            # make all estimations
            probas[voice_index] = models[voice_index].predict(batch_input_features,
                                                              batch_size=batch_size_per_voice) #softmax関数での出力
            #print(probas[voice_index])
            
            if not parallel_updates:
                # update
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    seq[time_indexes[voice_index][batch_index], voice_index] = pitch

        if parallel_updates:
            # update
            
            for voice_index in range(min_voice, num_voices):
                for batch_index in range(batch_size_per_voice):
                    #print(batch_index)
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch)  / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    seq[time_indexes[voice_index][batch_index], voice_index] = pitch


        ###################################################################################追加##メロディー：コード=1:2#####コード部分を反復
        for voice_index in range(num_voices-1, num_voices):
            batch_input_features = []
            time_indexes[voice_index] = []

            for batch_index in range(batch_size_per_voice):
                time_index = np.random.randint(timesteps, sequence_length + timesteps)
                time_index =  time_index -  time_index % 4###################################追加4の倍数のところだけサンプリング
                time_indexes[voice_index].append(time_index)
                #print(time_index)
                

                (left_feature,
                 central_feature,
                 right_feature,
                 label) = all_features(seq, voice_index, time_index, timesteps, num_pitches, num_voices)

                left_metas, central_metas, right_metas = all_metadatas(chorale_metadatas=extended_chorale_metas,
                                                                       metadatas=metadatas,
                                                                       time_index=time_index, timesteps=timesteps)
                left_local_seq, right_local_seq = make_local_sequences(seq, voice_index, time_index, num_pitches, note2indexes)

                input_features = {'left_features': left_feature[:, :],
                                  'central_features': central_feature[:],
                                  'right_features': right_feature[:, :],
                                  'left_metas': left_metas,
                                  'central_metas': central_metas,
                                  'right_metas': right_metas,
                                  'left_local_seqs':left_local_seq[:, :],
                                  'right_local_seqs':right_local_seq[:, :]}
                #print('='*10, voice_index, '='*10)
                #print(input_features)

                # list of dicts: predict need dict of numpy arrays
                batch_input_features.append(input_features)
            
            # convert input_features
            batch_input_features = {key: np.array([input_features[key] for input_features in batch_input_features])
                                    for key in batch_input_features[0].keys()
                                    }
            #print(batch_input_features)
            
            # make all estimations
            probas[voice_index] = models[voice_index].predict(batch_input_features,
                                                              batch_size=batch_size_per_voice) #softmax関数での出力
            #print(probas[voice_index])
            
            if not parallel_updates:
                # update
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    seq[time_indexes[voice_index][batch_index], voice_index] = pitch

        if parallel_updates:
            # update
            
            for voice_index in range(num_voices-1, num_voices):
                for batch_index in range(batch_size_per_voice):
                    #print(batch_index)
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch)  / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    seq[time_indexes[voice_index][batch_index], voice_index] = pitch
        ##########################################################################################ここまで追加
        

    return seq[timesteps:-timesteps, :]









# Utils
def load_model(model_name, yaml=True):
    """

    :rtype: object
    """
    if yaml:
        ext = '.yaml'
        model = model_from_yaml(open(model_name + ext).read())
    else:
        ext = '.json'
        model = model_from_json(open(model_name + ext).read())
    model.load_weights(model_name + '_weights.h5')
    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    print("model " + model_name + " loaded")
    return model


def save_model(model, model_name, yaml=True, overwrite=False):
    # SAVE MODEL
    if yaml:
        string = model.to_yaml()
        ext = '.yaml'
    else:
        string = model.to_json()
        ext = '.json'
    open(model_name + ext, 'w').write(string)
    model.save_weights(model_name + '_weights.h5', overwrite=overwrite)
    print("model " + model_name + " saved")


def create_models(model_name=None, create_new=False, num_dense=200, num_units_lstm=[200, 200],
                  pickled_dataset=BACKNUMBER_DATASET, num_voices=4, metadatas=None, timesteps=16):
    """
    Choose one model
    :param model_name:
    :return:
    """

    _, _, _, index2notes, _, _,_ = pickle.load(open(pickled_dataset, 'rb'))
  # X, X_metadatas, voice_ids, index2notes, note2indexes, metadata = pickle.load(open(pickled_dataset, 'rb'))
    #print('統計物理重い'*50)
    #print('index2notes===================\n', index2notes)
    num_pitches = list(map(len, index2notes)) #各パートの音の種類数
    #print('num_pitches=',num_pitches)
    for voice_index in range(num_voices):
        print('='*10 + 'パート',voice_index ,'='*10,timesteps)
        # We only need one example for features dimensions
        gen = generator_from_raw_dataset(batch_size=1, timesteps=timesteps, voice_index=voice_index,
                                         pickled_dataset=pickled_dataset)
        #batch_size=1としているのはgenerator_from_raw_datasetのwhile文の反復を１回にするため、ここではモデルのサイズさえわかれば良いので

        (
            (left_features,
             central_features,
             right_features),
            (left_metas, central_metas, right_metas),
            labels,left_local_seqs,right_local_seqs) = next(gen)

        """
        print()
        print('num_features_lr=', left_features.shape[-1])
        print('num_features_c=', central_features.shape[-1])
        print('num_pitches=', num_pitches)
        print('num_features_meta=', left_metas.shape[-1])
        print('num_dense=', num_dense)
        print('num_units_lstm=', num_units_lstm)
        print()
        """
        #if voice_index == 0:
            #np.set_printoptions(threshold=np.inf)
            #print('bbbbbbbbbb',left_features.shape)
            #print(left_features)
        if 'deepbach' in model_name:
            
            #model = deepBach(num_features_lr=left_features.shape[-1],
            #                 num_features_c=central_features.shape[-1],
            #                 num_pitches=num_pitches[voice_index],
            #                 num_features_meta=left_metas.shape[-1],
            #                 num_dense=num_dense, num_units_lstm=num_units_lstm,
            #                 timesteps = timesteps)

            model = deepBach_chord(num_features_lr=left_features.shape[-1],
                             num_features_c=central_features.shape[-1],
                             num_pitches=num_pitches[voice_index],
                             num_features_meta=left_metas.shape[-1],
                             num_dense=num_dense, num_units_lstm=num_units_lstm,
                             timesteps = timesteps,
                             num_localseqs_lr=left_local_seqs.shape[-1]
                             )
        elif 'skip' in model_name:
            model = deepbach_skip_connections(num_features_lr=left_features.shape[-1],
                                              num_features_c=central_features.shape[-1],
                                              num_features_meta=left_metas.shape[-1],
                                              num_pitches=num_pitches[voice_index],
                                              num_dense=num_dense, num_units_lstm=num_units_lstm, timesteps=timesteps)
        else:
            raise ValueError

        model_path_name = 'models/' + model_name + '_' + str(voice_index)
        if not os.path.exists(model_path_name + '.json') or create_new:
            save_model(model, model_name=model_path_name, overwrite=create_new)


def load_models(model_base_name=None, num_voices=4):
    """
    load 4 models whose base name is model_base_name
    models must exist
    :param model_base_name:
    :return: list of num_voices models
    """
    models = []
    for voice_index in range(num_voices):
    #for voice_index in range(num_voices)
        model_path_name = 'models/' + model_base_name + '_' + str(voice_index)
        model = load_model(model_path_name)
        model.compile(optimizer='adam', loss={'pitch_prediction': 'categorical_crossentropy'
                                              },
                      metrics=['accuracy'])
        models.append(model)
    return models


def train_models(model_name, steps_per_epoch, num_epochs, validation_steps, timesteps, pickled_dataset=BACKNUMBER_DATASET,
                 num_voices=4, batch_size=16, metadatas=None):
    """
    Train models
    :param batch_size:
    :param metadatas:

    """
    models = []
    for voice_index in range(num_voices):
        # Load appropriate generators

        #print('髪切ったー'*100)
        generator_train = (({'left_features': left_features,
                             'central_features': central_features,
                             'right_features': right_features,
                             'left_metas': left_metas,
                             'right_metas': right_metas,
                             'central_metas': central_metas,
                             'left_local_seqs':left_local_seqs,
                             'right_local_seqs':right_local_seqs
                             },
                            {'pitch_prediction': labels})
                           for (
                               (left_features, central_features, right_features),
                               (left_metas, central_metas, right_metas),
                                   labels,left_local_seqs,right_local_seqs)

                           in generator_from_raw_dataset(batch_size=batch_size, timesteps=timesteps,
                                                         voice_index=voice_index,
                                                         phase='train',
                                                         pickled_dataset=pickled_dataset
                                                         ))    ########訓練データ
        #print('春服買いたい'*100)

        
        generator_val = (({'left_features': left_features,
                           'central_features': central_features,
                           'right_features': right_features,
                           'left_metas': left_metas,
                           'right_metas': right_metas,
                           'central_metas': central_metas,
                           'left_local_seqs':left_local_seqs,
                             'right_local_seqs':right_local_seqs
                           },
                          {'pitch_prediction': labels})
                         for (
                             (left_features, central_features, right_features),
                             (left_metas, central_metas, right_metas),
                             labels,left_local_seqs,right_local_seqs)

                         in generator_from_raw_dataset(batch_size=batch_size, timesteps=timesteps,
                                                       voice_index=voice_index,
                                                       phase='test',
                                                       pickled_dataset=pickled_dataset
                                                       ))      #######検証データ

        #print('いちご食べたい'*100)
        #print(model_name)
        model_path_name = 'models/' + model_name + '_' + str(voice_index)

        model = load_model(model_path_name)

        model.compile(optimizer='adam', loss={'pitch_prediction': 'categorical_crossentropy'
                                              },
                      metrics=['accuracy']) #############すべてを理解した(*ﾉω・*)ﾃﾍ

        #print(validation_steps)
        model.fit_generator(generator_train, samples_per_epoch=steps_per_epoch,
                            epochs=num_epochs, verbose=1, validation_data=generator_val,
                            validation_steps=validation_steps)

        models.append(model) #modelという要素をもつリストを作った

        save_model(model, model_path_name, overwrite=True)
        #print('はい'*100)
        #print(len(models))
    return models





def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', help="model's range (default: %(default)s)",
                        type=int, default=16)
    parser.add_argument('-b', '--batch_size_train',
                        help='batch size used during training phase (default: %(default)s)',
                        type=int, default=128)
    parser.add_argument('-s', '--steps_per_epoch',
                        help='number of steps per epoch (default: %(default)s)',
                        type=int, default=500)
    parser.add_argument('--validation_steps',
                        help='number of validation steps (default: %(default)s)',
                        type=int, default=20)
    parser.add_argument('-u', '--num_units_lstm', nargs='+',
                        help='number of lstm units (default: %(default)s)',
                        type=int, default=[200, 200])
    parser.add_argument('-d', '--num_dense',
                        help='size of non recurrent hidden layers (default: %(default)s)',
                        type=int, default=200)
    parser.add_argument('-n', '--name',
                        help='model name (default: %(default)s)',
                        choices=['deepbach', 'skip'],
                        type=str, default='deepbach')
    parser.add_argument('-i', '--num_iterations',
                        help='number of gibbs iterations (default: %(default)s)',
                        type=int, default=6000)
    parser.add_argument('-t', '--train', nargs='?',
                        help='train models for N epochs (default: 15)',
                        default=0, const=15, type=int)
    parser.add_argument('-p', '--parallel', nargs='?',
                        help='number of parallel updates (default: 16)',
                        type=int, const=16, default=1)
    parser.add_argument('--overwrite',
                        help='overwrite previously computed models',
                        action='store_true')
    parser.add_argument('-m', '--midi_file', nargs='?',
                        help='relative path to midi file',
                        type=str, const='datasets/god_save_the_queen.mid')
    parser.add_argument('-l', '--length',
                        help='length of unconstrained generation',
                        type=int, default=256)
    parser.add_argument('--ext',
                        help='extension of model name',
                        type=str, default='')
    parser.add_argument('-o', '--output_file', nargs='?',
                        help='path to output file',
                        type=str, default='', const='generated_examples/example.mid')
    parser.add_argument('-r', '--reharmonization', nargs='?',
                        help='reharmonization of a melody from the corpus identified by its id',
                        type=int)
    args = parser.parse_args()
    print(args)

    # fixed set of metadatas to use when CREATING the dataset
    # Available metadatas:
    # metadatas = [FermataMetadatas(), KeyMetadatas(window_size=1), TickMetadatas(SUBDIVISION), ModeMetadatas()]
    metadatas = [TickMetadatas(SUBDIVISION), FermataMetadatas(), KeyMetadatas(window_size=1)]

    
    if args.ext:
        ext = '_' + args.ext
    else:
        ext = ''

    # datasets
    # set pickled_dataset argument


    #dataset_path = None
    pickled_dataset = BACKNUMBER_DATASET
    if not os.path.exists(pickled_dataset):
        initialization(metadatas=metadatas,voice_ids=[0, 1])


        
    # load dataset
    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas, chordname2chordset = pickle.load(open(pickled_dataset,
                                                                                       'rb'))
    #index2notes：サイズがパート数の配列、各要素はset型、setの中身は　各パートに出てきた音と番号の対応表
    
    print('index2notes' + '='*50)
    print(index2notes)
    #print('X'+'='*50)
    #print(X[0][1])

    #print('X_metadatas'+'='*50)
    #chorale = music21.corpus.parse('bach/bwv86.6')
    #chorale.show()
    #length = int(chorale.duration.quarterLength * SUBDIVISION)
    #print(length)
    #print(np.array(list(map(lambda x: x % 5,range(length)))))
    #print(type(X_metadatas))
    
    #TickMetadatas(X_metadatas).evaluate(chorale)
          
    NUM_VOICES = len(voice_ids
                     )
    num_pitches = list(map(len, index2notes)) #パートごとの全曲の音の数(+END,STARTとか)、サイズは(パート数)
    timesteps = args.timesteps
    batch_size = args.batch_size_train
    steps_per_epoch = args.steps_per_epoch
    validation_steps = args.validation_steps
    num_units_lstm = args.num_units_lstm
    model_name = args.name.lower() + ext
    sequence_length = args.length
    batch_size_per_voice = args.parallel
    num_units_lstm = args.num_units_lstm
    num_dense = args.num_dense


    #print('num_pitches = ', list(map(len, index2notes)))
    #print('NUM_VOICES = ', len(voice_ids))
    
    
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = None

    # when reharmonization
    if args.midi_file:
        melody = converter.parse(args.midi_file)
        melody = melody.transpose(2)
        melody = melody_to_inputs(melody.parts[0],int(melody.parts[0].duration.quarterLength*SUBDIVISION), index2note=index2notes[0], note2index=note2indexes[0])
        num_voices = NUM_VOICES - 1
        sequence_length = len(melody)
        # todo find a way to specify metadatas when reharmonizing a given melody
        chorale_metas = [metas.generate(sequence_length) for metas in metadatas]

    elif args.reharmonization:
        melody = X[args.reharmonization][0, :]
        num_voices = NUM_VOICES - 1
        chorale_metas = X_metadatas[args.reharmonization]
    else:
        num_voices = NUM_VOICES
        melody = None
        # todo find a better way to set metadatas

        # chorale_metas = [metas[:sequence_length] for metas in X_metadatas[11]]
        chorale_metas = [metas.generate(sequence_length) for metas in metadatas]#[TickMetadatas(SUBDIVISION), FermataMetadatas(), KeyMetadatas(window_size=1)]の各要素について.generateを実行、長さ=length(=160)のarray×3の配列を作る
    #print('sequence_length=', sequence_length )
    #print('choral_metas=', chorale_metas)
        
    num_iterations = args.num_iterations // batch_size_per_voice // num_voices #iteration回数を各パートで分配
    print('args.num_iterations=', args.num_iterations)
    print('batch_size_per_voice=', batch_size_per_voice)
    print('num_voices=', num_voices)
    parallel = batch_size_per_voice > 1
    print(' parallel=',  parallel)
    train = args.train > 0
    num_epochs = args.train #-t で指定した値
    print(' num_epochs=',  num_epochs)
    overwrite = args.overwrite

    #print(timesteps)
    if not os.path.exists('models/' + model_name + '_' + str(NUM_VOICES - 1) + '.yaml'):
        create_models(model_name, create_new=overwrite, num_units_lstm=num_units_lstm, num_dense=num_dense,
                      pickled_dataset=pickled_dataset, num_voices=num_voices, metadatas=metadatas, timesteps=timesteps)
    if train:
        models = train_models(model_name=model_name, steps_per_epoch=steps_per_epoch, num_epochs=num_epochs,
                              validation_steps=validation_steps, timesteps=timesteps, pickled_dataset=pickled_dataset,
                              num_voices=NUM_VOICES, metadatas=metadatas, batch_size=batch_size)
    else:
        models = load_models(model_name, num_voices=NUM_VOICES)
    temperature = 1.
    timesteps = int(models[0].input[0]._keras_shape[1])


    print('==============generation==============')
    seq = generation(model_base_name=model_name, models=models,
                     timesteps=timesteps,
                     melody=melody, initial_seq=None, temperature=temperature,
                     chorale_metas=chorale_metas, parallel=parallel, batch_size_per_voice=batch_size_per_voice,
                     num_iterations=num_iterations,
                     sequence_length=sequence_length,
                     output_file=output_file,
                     pickled_dataset=pickled_dataset)

    

if __name__ == '__main__':
    main()
