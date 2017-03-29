import argparse  # Command line parsing
import configparser  # Saving the models parameters
import datetime  # Chronometer
import os  # Files management
from tqdm import tqdm  # Progress bar
import tensorflow as tf

from textdata import TextData
from model import Model
import time
from random import randint


class Chatbot:
    def __init__(self):
        """
        """
        # Model/dataset parameters
        self.args = None

        # Task specific object
        self.textData = None  # Dataset
        self.model = None  # Sequence to sequence model

        # Tensorflow utilities for convenience saving/logging
        self.writer = None
        self.saver = None
        self.modelDir = ''  # Where the model is saved
        self.globStep = 0  # Represent the number of iteration for the current model

        # TensorFlow main session (we keep track for the daemon)
        self.sess = None

        # Filename and directories constants
        self.MODEL_DIR_BASE = 'save/model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.CONFIG_VERSION = '0.3'
        self.TEST_IN_NAME = 'data/test/samples.txt'
        self.TEST_OUT_SUFFIX = '_predictions.txt'
        self.SENTENCES_PREFIX = ['Q: ', 'A: ']

    def main(self, args=None):
        self.args = {}
        self.args['rootDir'] = os.getcwd()  # Use the current working directory
        self.args['corpus'] = 'cornell'
        self.args['maxLength'] = 10
        self.args['hiddenSize'] = 256
        self.args['numLayers'] = 2
        self.args['embeddingSize'] = 32
        self.args['softmaxSamples'] = 0
        self.args['numEpochs'] = 50
        self.args['saveEvery'] = 5000
        self.args['batchSize'] = 10
        self.args['learningRate'] = 0.001
        test_yes = True
        self.args['interactive'] = test_yes
        self.args['test'] = test_yes
        self.args['reset'] = False
        self.loadModelParams()  # Update the self.modelDir and self.globStep, for now, not used when loading Model (but need to be called before _getSummaryName)

        self.textData = TextData(self.args)

        self.model = Model(self.args, self.textData)

        # Saver/summaries
        self.writer = tf.train.SummaryWriter(self.modelDir)
        if '12' in tf.__version__:  # HACK: Solve new tf Saver V2 format
            self.saver = tf.train.Saver(max_to_keep=200, write_version=1)  # Arbitrary limit ?
        else:
            self.saver = tf.train.Saver(max_to_keep=200)

        self.sess = tf.Session()
        print('Initialize variables...')
        self.sess.run(tf.initialize_all_variables())
        self.managePreviousModel(self.sess)
        if self.args['interactive']:
            self.mainTestInteractive(self.sess)
        else:
            self.mainTrain(self.sess)
        self.sess.close()
    
    def set_up_things(self, args=None):
        self.args = {}
        self.args['rootDir'] = os.getcwd()  # Use the current working directory
        self.args['corpus'] = 'cornell'
        self.args['maxLength'] = 10
        self.args['hiddenSize'] = 256
        self.args['numLayers'] = 2
        self.args['embeddingSize'] = 32
        self.args['softmaxSamples'] = 0
        self.args['numEpochs'] = 50
        self.args['saveEvery'] = 5000
        self.args['batchSize'] = 10
        self.args['learningRate'] = 0.001
        self.args['reset'] = False
        test_yes = True
        self.args['interactive'] = test_yes
        self.args['test'] = test_yes
        self.loadModelParams()  # Update the self.modelDir and self.globStep, for now, not used when loading Model (but need to be called before _getSummaryName)
        self.textData = TextData(self.args)
        self.model = Model(self.args, self.textData)
        self.writer = tf.train.SummaryWriter(self.modelDir)
        if '12' in tf.__version__:  # HACK: Solve new tf Saver V2 format
            self.saver = tf.train.Saver(max_to_keep=200, write_version=1)  # Arbitrary limit ?
        else:
            self.saver = tf.train.Saver(max_to_keep=200)
        self.sess = tf.Session()
        print('Initialize variables...')
        self.sess.run(tf.initialize_all_variables())
        self.managePreviousModel(self.sess)
    
    def get_answer(self, ques):
        questionSeq = []  # Will be contain the question as seen by the encoder
        output_answer = ""
        answer = self.singlePredict(ques, questionSeq)
        if not answer:
            output_answer = "Woah buddy, slow down! Can you enter a few less words?"
        output_answer = self.textData.sequence2str(answer, clean=True)
        return output_answer

    def mainTrain(self, sess):

        mergedSummaries = tf.merge_all_summaries()  # Define the summary operator (Warning: Won't appear on the tensorboard graph)
        if self.globStep == 0:  # Not restoring from previous run
            self.writer.add_graph(sess.graph)  # First time only

        print('Start training (press Ctrl+C to save and exit)...')

        try:  # If the user exit while training, we still try to save the model
            for e in range(0, self.args['numEpochs']):

                print()
                print("----- Epoch {}/{} ; (lr={}) -----".format(e+1, self.args['numEpochs'], self.args['learningRate']))

                batches = self.textData.getBatches()

                # TODO: Also update learning parameters eventually

                tic = datetime.datetime.now()
                for nextBatch in tqdm(batches, desc="Training"):
                    # Training pass
                    ops, feedDict = self.model.step(nextBatch)
                    assert len(ops) == 2  # training, loss
                    _, loss, summary = sess.run(ops + (mergedSummaries,), feedDict)
                    self.writer.add_summary(summary, self.globStep)
                    self.globStep += 1

                    # Checkpoint
                    if self.globStep % self.args['saveEvery'] == 0:
                        self._saveSession(sess)

                toc = datetime.datetime.now()

                print("Epoch finished in {}".format(toc-tic))  # Warning: Will overflow if an epoch takes more than 24 hours, and the output isn't really nicer
        except (KeyboardInterrupt, SystemExit):  # If the user press Ctrl+C while testing progress
            print('Interruption detected, exiting the program...')

        self._saveSession(sess)  # Ultimate saving before complete exit


    def mainTestInteractive(self, sess):
        print('Testing: Launch interactive mode:')
        #allq = []
        #f2 = open('data/test/responses.txt', 'w')
        #with open('data/test/samples.txt') as f:
        #   allq = f.readlines()
        #for question in allq:
        output_answer = "we know no king but the king"
        i = 0
        while True:
            question = output_answer #input(self.SENTENCES_PREFIX[0])
            i = i + 1
            if i == 3:
                i = 1
            if len(question) >= 4:
                x = len(question) - randint(0,3)
            else:
                x = len(question) - randint(0,1)
            print("BOT" + str(i) + ": " + question)
            question = output_answer[:x]
            print()
            if question == '' or question == 'exit':
                break

            questionSeq = []  # Will be contain the question as seen by the encoder
            answer = self.singlePredict(question, questionSeq)
            if not answer:
                print('Warning: sentence too long, sorry. Maybe try a simpler sentence.')
                output_answer = output_answer[:-1]
                continue  # Back to the beginning, try again
            
            #f2.write("Q: " + question + " | ")
            #f2.write("A: " + self.textData.sequence2str(answer, clean=True) + "\n\n")
            #print('{}{}'.format(self.SENTENCES_PREFIX[1], self.textData.sequence2str(answer, clean=True)))
            output_answer = self.textData.sequence2str(answer, clean=True)
            time.sleep(2)
            #print("BOT2: " + output_answer)
            #print()
        #f2.close()

    def singlePredict(self, question, questionSeq=None):
        batch = self.textData.sentence2enco(question)
        if not batch:
            return None
        if questionSeq is not None:  # If the caller want to have the real input
            questionSeq.extend(batch.encoderSeqs)

        # Run the model
        ops, feedDict = self.model.step(batch)
        output = self.sess.run(ops[0], feedDict)  # TODO: Summarize the output too (histogram, ...)
        #print("OUTPUT: ")
        #print(output)
        answer = self.textData.deco2sentence(output)
        return answer


    def _saveSession(self, sess):
        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        self.saveModelParams()
        self.saver.save(sess, self._getModelName())  # TODO: Put a limit size (ex: 3GB for the modelDir)
        tqdm.write('Model saved.')

    def loadModelParams(self):
        self.modelDir = os.path.join(self.args['rootDir'], self.MODEL_DIR_BASE)
        # If there is a previous model, restore some parameters
        configName = os.path.join(self.modelDir, self.CONFIG_FILENAME)
        if os.path.exists(configName):
            # Loading
            config = configparser.ConfigParser()
            config.read(configName)

            # Restoring the the parameters
            self.globStep = config['General'].getint('globStep')
            self.args['maxLength'] = config['General'].getint('maxLength')  # We need to restore the model length because of the textData associated and the vocabulary size (TODO: Compatibility mode between different maxLength)
            #self.args.watsonMode = config['General'].getboolean('watsonMode')
            #self.args.datasetTag = config['General'].get('datasetTag')

            self.args['hiddenSize'] = config['Network'].getint('hiddenSize')
            self.args['numLayers'] = config['Network'].getint('numLayers')
            self.args['embeddingSize'] = config['Network'].getint('embeddingSize')
            self.args['softmaxSamples'] = config['Network'].getint('softmaxSamples')

        # For now, not arbitrary  independent maxLength between encoder and decoder
        self.args['maxLengthEnco'] = self.args['maxLength']
        self.args['maxLengthDeco'] = self.args['maxLength'] + 2

    def saveModelParams(self):
        """ Save the params of the model, like the current globStep value
        Warning: if you modify this function, make sure the changes mirror loadModelParams
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['version']  = self.CONFIG_VERSION
        config['General']['globStep']  = str(self.globStep)
        config['General']['maxLength'] = str(self.args['maxLength'])
        #config['General']['watsonMode'] = str(self.args['watsonMode'])

        config['Network'] = {}
        config['Network']['hiddenSize'] = str(self.args['hiddenSize'])
        config['Network']['numLayers'] = str(self.args['numLayers'])
        config['Network']['embeddingSize'] = str(self.args['embeddingSize'])
        config['Network']['softmaxSamples'] = str(self.args['softmaxSamples'])

        # Keep track of the learning params (but without restoring them)
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['learningRate'] = str(self.args['learningRate'])
        config['Training (won\'t be restored)']['batchSize'] = str(self.args['batchSize'])

        with open(os.path.join(self.modelDir, self.CONFIG_FILENAME), 'w') as configFile:
            config.write(configFile)

    def _getModelName(self):
        modelName = os.path.join(self.modelDir, self.MODEL_NAME_BASE)
        # if self.args.keepAll:  # We do not erase the previously saved model by including the current step on the name
        #     modelName += '-' + str(self.globStep)
        return modelName + self.MODEL_EXT

    def managePreviousModel(self, sess):
        print('WARNING: ', end='')
        modelName = self._getModelName()

        if os.listdir(self.modelDir):
            #if self.args.reset:
            #    print('Reset: Destroying previous model at {}'.format(self.modelDir))
            # Analysing directory content
            if os.path.exists(modelName):  # Restore the model
                print('Restoring previous model from {}'.format(modelName))
                self.saver.restore(sess, modelName)  # Will crash when --reset is not activated and the model has not been saved yet
                print('Model restored.')
            else:  # No other model to conflict with (probably summary files)
                print('No previous model found, but some files found at {}. Cleaning...'.format(self.modelDir))  # Warning: No confirmation asked
                self.args['reset'] = True
            
            if self.args['reset']:
                fileList = [os.path.join(self.modelDir, f) for f in os.listdir(self.modelDir)]
                for f in fileList:
                    print('Removing {}'.format(f))
                    os.remove(f)
        else:
            print('No previous model found, starting from clean directory: {}'.format(self.modelDir))
