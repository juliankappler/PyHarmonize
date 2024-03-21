# PyHarmonize: Adding harmony lines to recorded melodies in Python

## About

To use this module, the user provides

* a wav file containing a melody,
* the key in which the melody is, and
* the scale degree(s) of the desired harmony.

The module then outputs a wav file which contains the original melody, together with the added harmony line(s).

We first give some examples, the <a href="#installation">installation instructions</a> are further below.

## Examples (with audio files)

We here provide three audio examples together with the code used to generate them. See the folder [examples/](examples/) for more detailed example notebooks. 

*Note that the embedded mp4 video files that contain the audio in the following are by default muted.*

### Example 1: Added third on a distorted electric guitar

In [this example](examples/guitar%20distorted%20-%20E%20major%20-%20example%201.ipynb) we add a harmony line a third above the input melody, which is played on a distorted electric guitar. Here are the input signal used, as well as the final result:

https://github.com/juliankappler/PyHarmonize/assets/37583039/91dd3f93-258b-46c6-b4b0-f3c3e9d0a450

https://github.com/juliankappler/PyHarmonize/assets/37583039/f9778079-b2de-4a9d-85c2-da6e031d5816


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

https://github.com/juliankappler/PyHarmonize/assets/37583039/5201b28b-02b1-4faa-b9a2-0e8fbd8d156a

https://github.com/juliankappler/PyHarmonize/assets/37583039/06ab9648-8754-49e5-8e80-384222d8f1c2


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


https://github.com/juliankappler/PyHarmonize/assets/37583039/9c139503-741a-426f-8972-b9af1bd1d572


To generate this output, we pass <i>scale_degrees = [-8, -6, 3, 5, 8, 10]</i>, which adds pitch shifted signals an octave lower (-8), the third one octave lower (-6), a third up (3), a fifth up (5), an octave up (8), and a third an octave higher (10).

### Example 3: Added third, fifth, and octave on a clean electric guitar

In [this example](examples/guitar%20clean%20-%20A%20major.ipynb) we add thirds, fifths, and octaves to a melody in A major, which is played on a clean electric guitar. Here are input and output files:




https://github.com/juliankappler/PyHarmonize/assets/37583039/a98c056d-db0d-48ec-a343-e0e02f109a15



https://github.com/juliankappler/PyHarmonize/assets/37583039/224e7457-1a8a-4e60-b20f-b194872e99d9




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
