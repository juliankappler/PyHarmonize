#!/usr/bin/env python

import numpy as np
import librosa
import soundfile as sf
from scipy import signal


class harmony_generator:
	def __init__(self,
					parameters={}):
		#
		self.chromatic_scale = ['C','C#','D','D#','E','F','F#',
									'G','G#','A','A#','B']
		self.N_notes = 12
		#
		self.key = 'C'        # default key
		self.mode = 'major'   # default mode
		#
		self.scale_degrees = []
		self.semitones = []
		#
		self.scale_degrees_relative_amplitudes = []
		self.semitones_relative_amplitudes = []
		#
		self.signal_loaded = False
		self.sampling_rate_set = False
		#
		self.time_series_with_pitches_created = False
		#
		self.verbose = True # default value for verbose
		#
		self.input_filename = 'not set'
		self.sr = 'not set' # sampling rate
		self.x = 'not set' # this variable will later store the audio signal
		self.output_filename = 'not set'
		self.channel = 0 # process left channel of stereo signal
		#
		self.stft_windows_per_second = 20
		self.project_non_scale_notes = True
		self.dt_threshold_activate = 0.1 # s = 100 ms
		self.dt_threshold_deactivate = 0.1 # s = 100 ms
		#
		#
		self.chromatic_scale_dictionary = {}
		for i,note in enumerate(self.chromatic_scale):
			self.chromatic_scale_dictionary[note] = i
		#
		self.chromatic_scale_dictionary_inverse = { \
					item:key for \
		 			key,item in (self.chromatic_scale_dictionary).items() }
		#
		#
		self.circle_of_fifths = self.get_circle_of_fifths(chromatic_scale = \
													self.chromatic_scale)
		#
		# mode shifts are in semitones
		self.mode_shifts = {'major':0,
						'ionian':0,
						'dorian':10,
						'phrygian':8,
						'lydian':7,
						'mixolydian':5,
						'aeolian':3,
						'minor':3,
						'locrian':1}
		# self.mode_shifts[mode] = n
		# means that a scale in a given key with 'mode',
		# has the same notes as the major scale that starts n semitones higher.
		# For example, the minor (aka aeolian) scale in the key of A has the
		# same notes as the major scale in C, and the note is reached by going
		# up three semitones from A.
		#
		self.enharmonic_equivalents = {'Db':'C#',
										'Eb':'D#',
										'Gb':'F#',
										'Fb':'E',
										'Ab':'G#',
										'Bb':'A#',
										'Cb':'B'}
		#
		self.set_parameters(parameters=parameters)


	def set_parameters(self,parameters={}):
		'''
		Change parameters of an existing instance of this class
		'''
		#
		try:
			self.verbose = parameters['verbose']
		except KeyError:
			pass
		#
		try:
			self.input_filename = parameters['input_filename']
		except KeyError:
			pass
		#
		try:
			self.sr = parameters['sampling_rate'] # in Hz
			self.sampling_rate_set = True
		except KeyError:
			pass
		#
		try:
			self.x = parameters['x']
			self.N_samples = len(self.x)
			self.signal_loaded = True
		except KeyError:
			pass
		#
		try:
			self.channel = int(parameters['channel'])
		except KeyError:
			pass
		#
		try:
			self.key = parameters['key']
			if 'b' in self.key:
				self.key = self.get_enharmonic_equivalent(self.key)
		except KeyError:
			pass
		#
		try:
			self.mode = parameters['mode']
		except KeyError:
			pass
		#
		self.check_key_and_mode(key=self.key,mode=self.mode)
		#
		try:
			self.scale_degrees = parameters['scale_degrees']
		except KeyError:
			pass
		#
		try:
			self.semitones = parameters['semitones']
		except KeyError:
			pass
		#
		try:
			self.scale_degrees_relative_amplitudes = \
							parameters['scale_degrees_relative_amplitudes']
		except KeyError:
			pass
		#
		try:
			self.semitones_relative_amplitudes = \
							parameters['semitones_relative_amplitudes']
		except KeyError:
			pass
		#
		try:
			self.stft_windows_per_second = parameters['stft_windows_per_second']
		except KeyError:
			pass
		#
		try:
			self.project_non_scale_notes = parameters['project_non_scale_notes']
		except KeyError:
			pass
		#
		try:
			self.dt_threshold_activate = parameters['dt_threshold_activate']
		except KeyError:
			pass
		#
		try:
			self.dt_threshold_deactivate = parameters['dt_threshold_deactivate']
		except KeyError:
			pass
		#
		# set time array
		if self.sampling_rate_set and self.signal_loaded:
			self.t = np.arange(self.N_samples)/self.sr # in s
			self.dt = self.t[1] - self.t[0]


	def get_parameters(self):
		'''
		Return parameters
		'''
		output_dictionary = {
			'key':self.key,
			'mode':self.mode,
			'scale_degrees':self.scale_degrees,
			'semitones':self.semitones,
			'scale_degrees_relative_amplitudes':self.scale_degrees_relative_amplitudes,
			'semitones_relative_amplitudes':self.semitones_relative_amplitudes ,
			'signal_loaded':self.signal_loaded,
			'sampling_rate_set':self.sampling_rate_set,
			'time_series_with_pitches_created':self.time_series_with_pitches_created,
			'verbose':self.verbose,
			'input_filename':self.input_filename,
			'sampling_rate':self.sr,
			'output_filename':self.output_filename,
			'channel':self.channel,
			'stft_windows_per_second':self.stft_windows_per_second,
			'project_non_scale_notes':self.project_non_scale_notes,
			'dt_threshold_activate':self.dt_threshold_activate,
			'dt_threshold_deactivate':self.dt_threshold_deactivate
		}
		return output_dictionary

	def check_key_and_mode(self,key,mode):
		'''
		check if provided key and mode are valid
		'''
		#
		if key not in self.chromatic_scale:
			raise RuntimeError("Key not recognized. Please use one of the"\
			+" following keys: {0}".format(self.chromatic_scale))
		if mode not in self.mode_shifts.keys():
			raise RuntimeError("Mode not recognized. Please use one of the"\
			+" following modes: {0}".format(list(self.mode_shifts.keys())))

	def get_circle_of_fifths(self,chromatic_scale):
		'''
		Construct circle of fifths from chromatic scale
		'''
		circle_of_fifths = []
		semitones_fifth = 7
		for i in range(self.N_notes):
			circle_of_fifths.append(
							chromatic_scale[(i*semitones_fifth)%self.N_notes]
									)
		return circle_of_fifths

	def get_enharmonic_equivalent(self,key):
		'''
		For a key with a "b" (flat), return the
		the corresponding key with a "#" (sharp)
		'''
		try:
			return self.enharmonic_equivalents[key]
		except KeyError:
			raise RuntimeError("Key not recognized. Please use one of the"\
				+" following keys: {0}".format(self.chromatic_scale))

	def get_major_scale(self,
						key='C'):
		'''
		Construct major scale for given key

		For example, for key = 'C' the function returns a list
			['C' , 'D', 'E', 'F', 'G', 'A', 'B']

		'''
		#
		# get position of current root note in circle of fifths
		for i,current_note in enumerate(self.circle_of_fifths):
			if current_note == key:
				initial_index = i
				break
			if i == self.N_notes - 1:
				raise RuntimeError("Could not recognize provided key " \
					+ "'{0}'.\nPlease pass one of the following".format(key) \
					+ "keys:\n{0}".format(self.circle_of_fifths))
		#
		# construct scale using circle of fifths
		output_scale = []
		for j in range(3):
			output_scale.append(
				self.circle_of_fifths[(initial_index + 2*j)%self.N_notes]
								)
		for j in range(4):
			output_scale.append(
				self.circle_of_fifths[(initial_index - 1 + 2*j)%self.N_notes]
								)
		#
		return output_scale


	def get_scale(self,
					key='C',
					mode='major',
					return_equivalent_major_scale=False,
					verbose=False):
		'''
		Construct scale for given key and mode

		For example, for key = 'C' and mode = 'major', this function
		returns a list
			['C' , 'D', 'E', 'F', 'G', 'A', 'B']
		which is the C major scale.

		If the mode is not set to "major", and if
			return_equivalent_major_scale == True
		then the equivalent major scale is returned.
		For example, for key = 'A' and mode = 'minor', the major scale
		with the same notes is the C major scale. This means that for
			- return_equivalent_major_scale = False the function returns
				['A', 'B', 'C' , 'D', 'E', 'F', 'G'],
			  and for
			- return_equivalent_major_scale = True the function returns
				['C' , 'D', 'E', 'F', 'G', 'A', 'B']

		'''
		#
		# get key of equivalent major scale
		# ("equivalent" here means "the scale contains the same notes")
		shift_in_semitones = self.mode_shifts[mode]
		for i,current_note in enumerate(self.chromatic_scale):
			if current_note == key:
				equivalent_major_key = \
						self.chromatic_scale[(i + shift_in_semitones)%self.N_notes]
		#
		#
		# get the equivalent major scale
		equivalent_major_scale = self.get_major_scale(key=equivalent_major_key)
		#
		# if we want to return the equivalent major scale, we are done
		if return_equivalent_major_scale:
			if verbose:
				print('For the key {0} in mode {1}, the major'.format(key,mode) \
			 + ' key with the same notes is {0}'.format(equivalent_major_key) \
			 + ', which has the scale {0}'.format(equivalent_major_scale))
			return equivalent_major_scale
		#
		# to obtain the scale "key" in mode "mode", we now roll the
		# equivalent major scale so that the first element in the list is
		# "key", i.e. so that the returned scale starts with the correct
		# root note
		for i,current_note in enumerate(equivalent_major_scale):
			if current_note == key:
				initial_index = i
		output_scale = equivalent_major_scale[initial_index:] \
				+ equivalent_major_scale[:initial_index]
		if verbose:
			print('For the key {0} in mode {1}, the'.format(key,mode) \
		 	+ ' corresponding scale is {0}'.format(output_scale))
		return output_scale


	def get_semitone_shift_dictionary(self,
									key=None,
									mode=None,
									scale_degree=3, # 3 = one third up
									):
		'''
		For given
			- key and mode (and associated scale), as well as given
			- scale degree,
		this function calculates for every note in the scale how many semitones
		one has to move up/down to obtain the relative note with the scale degree.

		Example:
		For the key "C" with mode "major", and a value scale_degree = 3,
		this function constructs the dictionary

		semitone_shift_dictionary = {'C': 4, 'D': 3, 'E': 3, 'F': 4,
										'G': 4, 'A': 3, 'B': 3}.

		Here, the value
			semitone_shift_dictionary['C'] = 4
		means that one has to go up for semitones to reach the note which is
		a third (c.f. scale_degree = 3) up in the key (in this case the note E).
		For the note C, this is a major third. Since
			semitone_shift_dictionary['D'] = 3
		if we go a third up from D within the C major scale, the resulting
		interval is a minor third (D -> F).

		Generally, for any key/mode and scale_gree, the dictionary
		"semitone_shift_dictionary" has either one or two different values.
		The function also returns
			- the number of values N_shifts that appear,  as well as
			- a dictionary shift_dictionary which maps the semitone intervals
			  to an enumeration.

		In our above example, there are the two values 3 and 4 in the dictionary
		"semitone_shift_dictionary", so that we have

			N_shifts = 2
			shift_dictionary = {3: 0, 4: 1}.

		'''
		#
		if key is None:
			key = self.key
		if 'b' in key:
			key = self.get_enharmonic_equivalent(key)
		if mode is None:
			mode = self.mode
		self.check_key_and_mode(key=key,mode=mode)
		#
		scale = self.get_scale(key=key,
							mode = mode,
							return_equivalent_major_scale=False)
		#
		N_notes_in_scale = len(scale)
		#
		if scale_degree == 0:
			scale_degree = 1
		#
		shift = scale_degree - np.sign(scale_degree)
		octaves_to_shift = shift//N_notes_in_scale
		shift = shift%N_notes_in_scale
		#
		shift_dictionary = {}
		#
		for i,first_note in enumerate(scale):
			#
			second_note = scale[(i+shift)%N_notes_in_scale]
			#
			for j,current_note in enumerate(self.chromatic_scale):
				if first_note == current_note:
					index_of_first_note_in_chromatic_scale = j
				if second_note == current_note:
					index_of_second_note_in_chromatic_scale = j
			#
			if index_of_first_note_in_chromatic_scale > index_of_second_note_in_chromatic_scale:
				index_of_second_note_in_chromatic_scale += self.N_notes
			#
			interval_in_semitones = index_of_second_note_in_chromatic_scale \
									- index_of_first_note_in_chromatic_scale
			#
			shift_dictionary[first_note] = interval_in_semitones \
											+ self.N_notes * octaves_to_shift
		#
		semitone_shift_dictionary = shift_dictionary
		#
		set_of_shift_semitones = set( val for val in semitone_shift_dictionary.values())
		N_shifts = len(set_of_shift_semitones)
		#
		shift_dictionary = {}
		for i,item in enumerate(set_of_shift_semitones):
			shift_dictionary[item] = i
		#
		#
		output_dictionary = {'semitone_shift_dictionary':semitone_shift_dictionary,
				'N_shifts':N_shifts,
				'shift_dictionary':shift_dictionary}
		#
		return output_dictionary


	def load_file(self,filename=None,
					channel=0):
		'''
		Load wav file

		This function is basically a wrapper for SoundFile.read()
		'''
		#
		if filename is None:
			if self.input_filename == 'not set':
				raise RuntimeError("No input filename provided")
			filename = self.input_filename
		#
		if channel is None:
			channel = self.channel
		#
		x, self.sr = sf.read(file=filename,always_2d=True)
		self.x = x[:,channel]
		self.N_samples = len(self.x)
		self.signal_loaded = True
		self.sampling_rate_set = True
		self.time_series_with_pitches_created = False
		#
		parameters = {'x':self.x,
						'sampling_rate':self.sr}
		self.set_parameters(parameters=parameters)


	def get_time_series_with_pitches(self,
									return_result=False,
									stft_windows_per_second=None,
									smoothing_duration=0.2, # 100 ms
									):
		'''

		To Do: improve documentation of this function

		For a given loaded audio signal series, this function calculates
		the dominant pitch at every (coarse grained) timestep, and returns
		an array with the corresponding lowest frequency at which this
		pitch appears

		'''
		#
		#
		if self.signal_loaded == False:
			raise RuntimeError("No audio signal loaded")
		if self.sampling_rate_set == False:
			raise RuntimeError("No sampling rate set")
		#
		if stft_windows_per_second is None:
			stft_windows_per_second = self.stft_windows_per_second
		#
		nperseg = int(self.sr/stft_windows_per_second)
		self.f, self.t_stft, self.Zxx = signal.stft(self.x,
										fs=self.sr,
										nperseg=nperseg,
										)
		self.N_t_stft = len(self.t_stft)
		self.dt_stft = self.t_stft[1]-self.t_stft[0]
		#
		pitches, magnitudes = librosa.piptrack(S =self.Zxx,
												sr=self.sr,
												threshold=0.1,
												fmin=10,fmax=5000)
		#
		self.pitch_stft = np.zeros(self.N_t_stft,dtype=float)
		for i,(p,m) in enumerate(zip(pitches.T,magnitudes.T)):
			self.pitch_stft[i] = self.get_dominant_pitch(pitches=p,
											magnitudes=m)
		#
		self.time_series_with_pitches_created = True
		#
		'''
		# c.f. https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
		window_len = int( smoothing_duration / self.dt_stft )
		print('window_len =',window_len)
		s=np.r_[self.pitch_stft[window_len-1:0:-1],
				self.pitch_stft,
				self.pitch_stft[-2:-window_len-1:-1]]
		w = np.hanning(window_len)
		self.pitch_stft=np.convolve(w/w.sum(),s,mode='valid')
		''';
		#
		if return_result:
			return {'t_stft':self.t_stft,
					'f':self.f,
					'Zxx':self.Zxx,
					'pitch_stft':self.pitch_stft}


	def get_dominant_pitch(self,
							pitches,magnitudes,
							weight_power=1.):
		'''
		To Do: Add documentation
		'''
		#
		note_weights = np.zeros(self.N_notes,dtype=float)
		#
		mask = (pitches > 0)
		pitches = pitches[mask]
		magnitudes = magnitudes[mask]
		for i,(pitch,magnitude) in enumerate(zip(pitches,magnitudes)):
			#
			note = librosa.hz_to_note(pitch,octave=False,unicode=False)
			note_weights[self.chromatic_scale_dictionary[note]] += magnitude**weight_power
		#
		detected_note = self.chromatic_scale_dictionary_inverse[np.argmax(note_weights)]
		#
		# get the first frequency where we have the maximal value of the current pitch
		for i,pitch in enumerate(pitches):
			if librosa.hz_to_note(pitch,
								octave=False,
								unicode=False) == detected_note:
				return pitch
		return 0

	def get_pitch_shift_characteristic_functions(self,
						semitone_shift_dictionary,
						shift_dictionary,
						N_shifts,
						scale,
						project_non_scale_notes=None):
		'''
		To Do: Add documentation
		'''
		#
		if project_non_scale_notes is None:
			project_non_scale_notes = self.project_non_scale_notes
		#
		N_width = int(np.ceil(self.dt_stft/self.dt/2))
		#
		characteristic_functions_nonsmooth = [np.zeros(self.N_samples,
												dtype=float) \
										for i in range(N_shifts)]
		#
		for i,pitch in enumerate(self.pitch_stft):
			if pitch != 0:
				note = librosa.hz_to_note(pitch,octave=False,unicode=False)
				if note not in scale:
					if project_non_scale_notes:
						hz_of_note = librosa.note_to_hz(note)
						# if a note is not in the scale, then both the notes a
						# semitone up and a semitone down are in the scale
						closest_notes_in_scale = np.array([2**(1./12.),
															2**(-1./12.)],
														dtype=float) \
											* hz_of_note
						#
						diff_array = np.log(closest_notes_in_scale) \
										- np.log( hz_of_note )
						index = np.argmin( np.fabs( diff_array  ) )
						#
						pitch_of_closest_note_in_scale = \
								closest_notes_in_scale[index]
						#
						note = librosa.hz_to_note(pitch_of_closest_note_in_scale,
										octave=False,unicode=False)
					else:
						continue
				#
				index_of_closest_time_in_full_resolution = \
							np.argmin(np.fabs(self.t - self.t_stft[i]))
				#
				current_shift = semitone_shift_dictionary[note]
				function_index = shift_dictionary[current_shift]
				if i == 0:
					i0 = 0
					i1 = N_width
				elif i == self.N_t_stft-1:
					i0 = self.N_samples - N_width
					i1 = self.N_samples
				else:
					i0 = index_of_closest_time_in_full_resolution - N_width
					i1 = index_of_closest_time_in_full_resolution + N_width
				characteristic_functions_nonsmooth[function_index][i0:i1] = 1.
		#

		N_threshold_activate = int(np.floor(self.dt_threshold_activate/self.dt))
		#
		self.dt_threshold_deactivate = 0.1 # s = 100 ms
		N_threshold_deactivate = int(np.floor(self.dt_threshold_deactivate/self.dt))
		#

		characteristic_functions = []
		for i,cf in enumerate(characteristic_functions_nonsmooth):
			#
			cf_smoothed = cf.copy()
			#
			where_cf_1 = np.where(cf==1)[0]
			d_where_cf_1 = where_cf_1[1:] - where_cf_1[:-1]
			#
			for j,d_indices in enumerate(where_cf_1):
				if d_indices > 1:
					if d_indices <= N_threshold_activate:
						cf_smoothed[where_cf_1[j]:where_cf_1[j+1]] = 1.
			#
			where_cf_0 = np.where(cf==0.)[0]
			d_where_cf_0 = where_cf_0[1:] - where_cf_0[:-1]
			#
			for j,d_indices in enumerate(d_where_cf_0):
				if d_indices > 1:
					if d_indices <= N_threshold_deactivate:
						#print(where_cf_0[j],where_cf_0[j+1],d_indices)
						cf_smoothed[where_cf_0[j]:where_cf_0[j+1]] = 0.
			characteristic_functions.append(cf_smoothed)
		#
		return characteristic_functions



	def add_harmonies(self,
					key=None,
					mode=None,
					scale_degrees=None,
					semitones=None,
					scale_degrees_relative_amplitudes=None,
					semitones_relative_amplitudes=None,
					output_filename = None,
					verbose=None):
		#
		'''
		To Do: Add documentation
		'''
		#
		#
		if key is None:
			key = self.key
		else:
			if 'b' in key:
				key = self.get_enharmonic_equivalent(key)
		#
		if mode is None:
			mode = self.mode
		#
		self.check_key_and_mode(key=key,mode=mode)
		#
		if scale_degrees is None:
			scale_degrees = self.scale_degrees
		if semitones is None:
			semitones = self.semitones
		if scale_degrees_relative_amplitudes is None:
			# if current self.scale_degrees_relative_amplitudes is inconsistent
			# with provided semitones, update self.scale_degrees_relative_amplitudes
			scale_degrees_relative_amplitudes = self.scale_degrees_relative_amplitudes
			if len(scale_degrees_relative_amplitudes) != len(scale_degrees):
				scale_degrees_relative_amplitudes = np.ones(len(scale_degrees))
		else:
			if len(self.scale_degrees_relative_amplitudes) != len(scale_degrees):
				scale_degrees_relative_amplitudes = np.ones(len(scale_degrees))

		if semitones_relative_amplitudes is None:
			# if current self.semitones_relative_amplitudes is inconsistent
			# with provided semitones, update self.scale_degrees_relative_amplitudes
			semitones_relative_amplitudes = self.semitones_relative_amplitudes
			if len(semitones_relative_amplitudes) != len(semitones):
				semitones_relative_amplitudes = np.ones(len(semitones))
		else:
			if len(self.semitones_relative_amplitudes) != len(semitones):
				semitones_relative_amplitudes = np.ones(len(semitones))
		if verbose is None:
			verbose = self.verbose
		#
		#
		# load file
		if self.signal_loaded == False:
			self.load_file()
		#
		# get trajectory with pitch
		if self.time_series_with_pitches_created == False:
			self.get_time_series_with_pitches()
		#
		# get scale for current key
		scale = self.get_scale(key=key,
						mode=mode,
						verbose=verbose)
		#
		# for every scale degree, create pitch-shifted signal
		scale_degrees_shifted_audio = []
		for i,scale_degree in enumerate(scale_degrees):
			#
			result_dictionary = self.get_semitone_shift_dictionary(key=key,
											mode=mode,
											scale_degree=scale_degree)
			semitone_shift_dictionary = result_dictionary['semitone_shift_dictionary']
			N_shifts = result_dictionary['N_shifts']
			shift_dictionary = result_dictionary['shift_dictionary']
			#
			characteristic_functions = self.get_pitch_shift_characteristic_functions(
								semitone_shift_dictionary=semitone_shift_dictionary,
								shift_dictionary=shift_dictionary,
								N_shifts=N_shifts,
								scale=scale)
			#
			current_shifted_audio = np.zeros(self.N_samples,dtype=float)
			#
			#
			for j,(shift_in_semitones,index) in enumerate(shift_dictionary.items()):
				current_shifted_audio += librosa.effects.pitch_shift(
										y=self.x,
										sr=self.sr,
										n_steps=shift_in_semitones) * \
										characteristic_functions[index]
			#
			scale_degrees_shifted_audio.append( current_shifted_audio )
		#
		# for every semitone shift, create pitch-shifted signal
		semitones_shifted_audio = []
		for i,current_semitones in enumerate(semitones):
			semitones_shifted_audio.append ( librosa.effects.pitch_shift(
									y=self.x,
									sr=self.sr,
									n_steps=current_semitones) )
		#
		# add pitch-shifted signals to input audio
		output_audio = (self.x).copy()
		for i,current_shifted_audio in enumerate(scale_degrees_shifted_audio):
			output_audio += scale_degrees_relative_amplitudes[i] \
												* current_shifted_audio
		for i, current_shifted_audio in enumerate(semitones_shifted_audio):
			output_audio += semitones_relative_amplitudes[i] \
												* current_shifted_audio
		#
		# save result
		if output_filename is None:
			if self.output_filename != 'not set':
				sf.write(file=self.output_filename,
							data=output_audio,samplerate=self.sr)
		else:
			sf.write(file=output_filename,
						data=output_audio,samplerate=self.sr)
		#
		# return result
		output_dictionary = {
					'x_in':self.x,
					'sampling_rate':self.sr,
					'x_out':output_audio, # full mix
					'scale_degrees_x':scale_degrees_shifted_audio, # individual
					'semitones_x':semitones_shifted_audio,         # harmomnies
					#
					'scale_degrees':scale_degrees,
					'semitones':semitones,
					#
					'scale_degrees_relative_amplitudes':scale_degrees_relative_amplitudes,
					'semitones_relative_amplitudes':semitones_relative_amplitudes,
					#
					'stft_windows_per_second':self.stft_windows_per_second,
					'project_non_scale_notes':self.project_non_scale_notes,
					'dt_threshold_activate':self.dt_threshold_activate,
					'dt_threshold_deactivate':self.dt_threshold_deactivate,
					}
		#
		return output_dictionary
