poetry run autowah --Q 8 --input_gain 3 --mix 0.0 -i "./03 Audio 3.wav" -o "03 dry.wav"
poetry run autowah --Q .707 --input_gain 3 --mix 1.0 -i "./03 Audio 3.wav" -o "03 wet low Q.wav"
poetry run autowah --Q 8 --input_gain 3 --gain .9 --mix 1.0 -i "./03 Audio 3.wav" -o "03 wet mid Q.wav"
poetry run autowah --Q 20 --input_gain 3 --gain .9 --mix 1.0 -i "./03 Audio 3.wav" -o "03 wet high Q.wav"

poetry run autowah --is_bandpass True --Q 10 --input_gain 3 --envelope_gain 2.5 --gain 5.0 --mix 1.0 -i "./03 Audio 3.wav" -o "03 wet bandpass mid Q.wav"
poetry run autowah --is_bandpass True --Q 10 --input_gain 3 --envelope_gain 2.5 --gain 5.0 --mix .5 -i "./03 Audio 3.wav" -o "03 mixed bandpass mid Q.wav"
