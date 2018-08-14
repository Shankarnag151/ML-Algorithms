import logging
import os
from flask import Flask, request, send_file
#from flask.ext.twisted import Twisted
from flask_twisted import Twisted
from deepspeech.model import Model
import scipy.io.wavfile as wav
from werkzeug.utils import secure_filename
from timeit import default_timer as timer
#from ffmpy import FFmpeg
from flask_cors import CORS, cross_origin



logging.basicConfig(filename='DownloadMeltWatersTwitterData.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DownloadMeltWatersTwitterData')
logger.addHandler(logging.StreamHandler())
logger.info("{} Starting ...".format(__name__))


UPLOAD_FOLDER = '/tmp/flask'
ALLOWED_EXTENSIONS = set(['wav', 'webm'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)
twisted = Twisted(app)

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


model = None

@app.route('/')
def hello_world():
    return 'Hello World!'



@app.route('/webmToWav', methods=['GET', 'POST'])
def convertToText():
    savedAudioPath = ''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            savedAudioPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(savedAudioPath)

    if len(savedAudioPath) == 0:
        return 'No File Found'
    else:
        fileNoExt = filename.split('.')[0]
        wavFile = os.path.join(app.config['UPLOAD_FOLDER'], fileNoExt+'.wav')
        # ff = FFmpeg(
        #     inputs={savedAudioPath: None},
        #     outputs={wavFile: '-y -acodec pcm_s16le -ac 1 -ar 16000'})
        # ff.run()
        os.system('ffmpeg -i '+savedAudioPath+' -y -acodec pcm_s16le -ac 1 -ar 16000 '+wavFile)

        return send_file(
            wavFile,
            mimetype="audio/wav",
            as_attachment=True,
            attachment_filename="test.wav")

@app.route('/wavToText', methods=['GET', 'POST'])
def wavToText():
    global model

    ## Read Audio file to local fs
    savedAudioPath = ''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
            #return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return 'No selected file'
            #return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            savedAudioPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(savedAudioPath)
            #return redirect(url_for('uploaded_file', filename=filename))
            #url = url_for('uploaded_file', filename=filename)
            #return url

    if len(savedAudioPath) == 0:
        return 'No File Found'

    ## Run Inference
    model_load_start = timer()
    fs, audio = wav.read(savedAudioPath)
    audio_length = len(audio) * ( 1 / 16000)
    res = model.stt(audio, fs)
    inference_end = timer() - model_load_start
    logger.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))

    return res


if __name__ == '__main__':
    ds = Model('E:/Udemy/deepspeech-0.1.1-models/models/output_graph.pb', N_FEATURES, N_CONTEXT, 'E:/Udemy/deepspeech-0.1.1-models/models/alphabet.txt',
               BEAM_WIDTH)

    ds.enableDecoderWithLM('E:/Udemy/deepspeech-0.1.1-models/models/alphabet.txt', 'E:/Udemy/deepspeech-0.1.1-models/models/lm.binary',
                           'E:/Udemy/deepspeech-0.1.1-models/models/trie', LM_WEIGHT, WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)

    model = ds

    # context = SSL.Context(SSL.SSLv23_METHOD)
    # context.use_privatekey_file(app.root_path+'/certs/key.pem')
    # context.use_certificate_file(app.root_path+'/certs/cert.pem')
    # context = ('cert.crt', 'key.key')
    #context = ('cert.crt', 'key.key')
    #app.run(port=9090, host= '0.0.0.0', ssl_context=(app.root_path+'/certs/cert.pem', app.root_path+'/certs/key.pem'), threaded=True)
    app.run(port=9090, host= '0.0.0.0')