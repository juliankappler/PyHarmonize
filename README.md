# PyHarmonize: Python module for adding harmony lines to recorded melodies

## About

To use this module, the user provides

* a wav file containing a melody,
* the key in which the melody is, and
* the scale degree(s) of the harmony.

The module then outputs a wav file which contains the original melody, together with the added harmony line(s).

We first give some examples, the <a href="#installation">installation instructions</a> are further below.

## Examples (with audio files)

We here provide three audio examples together with the code used to generate them. See the folder [examples/](examples/) for more detailed example notebooks. 

<sub><sup>**Note that the embedded mp4 audio files in the following are by default muted. In Safari, I have to click into the "video window" three times to see the button that allows me to unmute.**</sub></sup>

### Example 1: Added third on a distorted electric guitar

In [this example](examples/guitar%20distorted%20-%20E%20major%20-%20example%201.ipynb) we add a harmony line a third above the input melody, which is played on a distorted electric guitar. Here are the input signal used, as well as the final result:

https://user-images.githubusercontent.com/37583039/153066178-24fec761-7809-4b3d-890a-dfe4ecd7f2f3.mp4

https://user-images.githubusercontent.com/37583039/153066889-b353b151-7bb9-413c-92ae-6db3186da832.mp4

And here is the code used to generate this output:

```Python
import PyHarmonize

# Create dictionary with parameters
parameters = {'input_filename':'./guitar_distorted_E_major_ex1.wav', # input audio is in the key of E major
              'output_filename':'./guitar_distorted_E_major_ex1_with_harmony.wav',
              'key':'E',
              'mode':'major'}

# Generate instance of the class harmony_generator
harmony_generator = PyHarmonize.harmony_generator(parameters=parameters)

# Add harmony
# Note that scale_degrees = [3] means we add one melody line,
# which is always three notes higher within the scale. Depending on the note
# played, "three notes higher within the scale" is either 3 or 4 semitones up.
output_dictionary = harmony_generator.add_harmonies(scale_degrees = [3])
```

### Example 2: Added third and fifth on a distorted electric guitar

In [this example](examples/guitar%20distorted%20-%20E%20major%20-%20example%202.ipynb) we add two harmony lines to an input signal. Here are the input signal and the result:

https://user-images.githubusercontent.com/37583039/153067531-f237082f-3449-46b3-8528-bc1410881791.mp4

https://user-images.githubusercontent.com/37583039/153067555-340320e4-9759-4019-8e35-2bdce7c6229b.mp4

The [code for this example](examples/guitar%20distorted%20-%20E%20major%20-%20example%202.ipynb) is essentially the same as in the first example, except that now the list <i>scale_degrees</i> contains more than one element:

```Python
import PyHarmonize

# Create dictionary with parameters
parameters = {'input_filename':'./guitar_distorted_E_major_ex2.wav', # input audio is in the key of E major
              'output_filename':'./guitar_distorted_E_major_ex2_with_harmony.wav',
              'key':'E',
              'mode':'major'}

# Generate instance of the class harmony_generator
harmony_generator = PyHarmonize.harmony_generator(parameters=parameters)

# Add harmony
output_dictionary = harmony_generator.add_harmonies(scale_degrees = [3, 5]) # add third and fifth
```

If we add some more octaves and thirds, we can generate a more synthesizer-like sound. Here is an example for that:

https://user-images.githubusercontent.com/37583039/153067878-fa2d3d9e-2c5f-4bc9-b133-99574a63c8a5.mp4

To generate this output, we pass <i>scale_degrees = [-8, -6, 3, 5, 8, 10]</i>, which adds pitch shifted signals an octave lower (-8), the third one octave lower (-6), a third up (3), a fifth up (5), an octave up (8), and a third an octave higher (10).

### Example 3: Added third, fifth, and octave on a clean electric guitar

In [this example](examples/guitar%20clean%20-%20A%20major.ipynb) we add thirds, fifths, and octaves to a melody in A major, which is played on a clean electric guitar. Here are input and output files:

https://user-images.githubusercontent.com/37583039/153068671-23657df3-e2ac-4de9-bbf5-a57475e8c9f8.mp4


https://user-images.githubusercontent.com/37583039/153068718-106ed015-4dac-4762-a033-ab8815c5d7ea.mp4


The [code for generating this harmony](examples/guitar%20clean%20-%20A%20major.ipynb) is:

```Python
import PyHarmonize

# Create dictionary with parameters
parameters = {'input_filename':'./guitar_clean_A_major.wav', # input audio is in the key of A major
              'output_filename':'./guitar_clean_A_major_with_harmony.wav',
              'key':'A',
              'mode':'major'}

# Generate instance of the class harmony_generator
harmony_generator = PyHarmonize.harmony_generator(parameters=parameters)

# Add harmony
output_dictionary = harmony_generator.add_harmonies(scale_degrees = [3,5,8])
# The list
#       scale_degrees = [3, 5, 8]
# means that we add four melody lines:
# 1. a third up
# 2. a fifth up
# 3. one octave up
```

## <a id="installation">  Installation

To install the module PyHarmonize, as well as its requirements ([NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [librosa](https://librosa.org/), and [SoundFile](https://github.com/bastibe/python-soundfile)), clone this repository and run the installation script:

```bash
>> git clone https://github.com/juliankappler/PyHarmonize.git
>> cd PyHarmonize
>> pip install -r requirements.txt
>> python setup.py install
```
