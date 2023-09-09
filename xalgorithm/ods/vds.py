"""
pip install youtube-transcript-api
pip install spacy
python -m spacy download en_core_web_lg
"""
from youtube_transcript_api import YouTubeTranscriptApi as API
from youtube_transcript_api.formatters import Formatter
from xalgorithm import ojoin, ocode

LANG = 'en'
DIR  = ojoin("~", '.cache', 'subtitles', expand_user=True)

class TextFormatter(Formatter):
    def format_transcript(self, transcript, **kwargs):
        """Converts a transcript into plain text with no timestamps.
        """
        return '\n'.join(line['text'] for line in transcript)

    def format_transcripts(self, transcripts, **kwargs):
        """Converts a list of transcripts into plain text with no timestamps.
        """
        return '\n\n\n'.join([self.format_transcript(transcript, **kwargs) for transcript in transcripts])

def parse_transcript_by_spacy(text):
    global NLP
    doc = NLP(text)
    output = ""
    for sent in doc.sents:
        output += str(sent)+"\n"
    return output

def parse_vds(args):
    if args.spacy:
        import spacy
        global NLP
        NLP = spacy.load("en_core_web_lg")
    for video_id in args.video_ids:
        try:
            tlst = API.list_transcripts(video_id)
            if LANG in tlst._manually_created_transcripts:
                transcript = tlst.find_manually_created_transcript([LANG])
            elif LANG in tlst._generated_transcripts:
                transcript = tlst.find_generated_transcript([LANG])
            else:
                raise Exception
        except Exception:
            raise RuntimeError("cannot find subtitles for this video {}".format(video_id))
        transcript = TextFormatter().format_transcript(transcript.fetch())
        transcript_path = ojoin(DIR, video_id)
        if args.spacy:
            transcript = parse_transcript_by_spacy(transcript)
        with open(transcript_path, 'w') as file: 
            file.write(transcript)
        ocode(transcript_path)