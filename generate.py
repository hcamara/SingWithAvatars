import os
import json
from libs.sounds import Sounds
from libs.pattern import Pattern
from libs.utils import Utils
from pydub import AudioSegment

# Init vars
Utils, combined = [Utils(), AudioSegment.empty()]
kb4pattern, data = [[], []]

# Load config
config = json.load(open('config.json'))

# Load Acoustic
knowledgeBase = json.load(open(config['acoustic'] + config['acoustic_data']))

# Load Testing data
with open(config['datasets'] + config['testing']) as f:
    forTesting = f.readlines()
forTesting =  [line.strip() for line in forTesting] 

# Sounds extraction
for ech in knowledgeBase:
    kb4pattern.append(ech['predicted_label'])

pattern = Pattern(kb4pattern)
sounds = Utils.correlate(pattern.matching(forTesting), knowledgeBase)

# Song construction
for sound in sounds:
    audio = AudioSegment.from_file(config['acoustic'] + config['forground'] + 
                                sound + config['ext'], format=(config['ext'])[1:])
    # Detect and delete silences
    start = Utils.detect_silence(audio)
    end = Utils.detect_silence(audio.reverse())  
    audio = audio[start:len(audio) - end]

    # Concat audio
    combined += audio

# Export temp audio file   
combined.export(config['audio'] + config['ext'], format=(config['ext'])[1:])

# Mixed forground and background 
combined = (AudioSegment.from_file(
            config['audio'] + config['ext'])).overlay(
                AudioSegment.from_file(config['acoustic'] + config['background'] + config['ext']))

# Export temp audio file  
combined.export(config['audio'] + "/tmp" + config['ext'], 
                format=(config['ext'])[1:])

# Remove phase
sound = Sounds(config['audio'] + "/tmp" + config['ext'])
sound.write(config['audio'] + '/' + config['audio'] + config['ext'], 
            ((sound).removePhase()))

# unlinks
os.unlink(config['audio'] + config['ext'])
os.unlink(config['audio'] + "/tmp" + config['ext'])