
class Utils(object):
    def detect_silence(self, sound, silence_threshold=-50.0, chunk_size=10):
        trim_ms = 0
        assert chunk_size > 0
        while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
            trim_ms += chunk_size
        return trim_ms

    def correlate(self, sounds, knowledgeBase):
        final_sounds = []
        for sound in sounds:
            for kb in knowledgeBase:
                if kb['predicted_label'] == sound:
                    final_sounds.append(kb['name'])
        return final_sounds