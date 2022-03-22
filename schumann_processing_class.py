import math
import numpy as np
from scipy.fftpack import fft
import math
import h5py
from lmfit import minimize, Parameters

class Schumann_processing:
# Protected variables
    #Set input field data threshold
    __signal_threshold = 9999.0 # time series threshold in mV
    
    # Set registration data parameters (change if needed)
    __sampling_frequency = 256   # sampling frequency of a monitoring station
    __time_bulk = 12*60 # [seconds] Is a statically significant time interval.
    __window_width = 8  # [seconds] shows window width for each 12 minute interval

    # Set samples in window (Modified window width for FFT)
    __fft_window_sampling = 2**11 # Calculate_number_of_samples_for_fft(window_width*sampling_frequency)
    
    # Standart resonance parameters
    # [a1, a2, a3, f1, f2, f3, s1, s2, s3] ('ai' for i-th amplitude, 'fi' for i-th frequency, 'si' for i-th resonance peak width, m and n for linear term)
    __standart_resonance_parameters = np.array([0.2,0.2,0.2, 8.011,14.2,20.63, 1.78,1.94,2.56])
    
    # Spectrum part for the fitting procedure, which contains three first resonance frequencies
    __f_min = 6
    __f_max = 24

    # Set spectral boundaries for h5 output format
    __h5_low_freq = 5
    __h5_up_freq = 40
    __h5_output_paramter = 1

    # Calibration file
    __calibration_file = ''
    ######################################################################
    
    # Processing for a month file
   # def Process_average_data(self, names, calibration_name=''):
   #     # Set calibration file
   #     self.__calibration_file = calibration_name
   #     # Calculate parameters of windows and intervals
   #     self.__Calculate_Mesh_Values()
   #     # Load txt file with field data
   #     data = self.__Load_txt_File(name[0])
   # 
   #     for w in names:


    def Set_Calibration_File(self, calibration_name):
        # Set calibration file
        self.__calibration_file = calibration_name

    # Processing for a day file
    def Process_txt_day_data(self, name, month_name=''):
        # Set month average SR file
        self.__month_file_path = month_name
        # Calculate parameters of windows and intervals
        self.__Calculate_Mesh_Values()
        
        # Check window parameters
        self.__Check_Window_Sampling()
        
        # Load txt file with field data
        data = self.__Load_txt_File(name)
        
        self.__Initialize_Arrays_For_FFT_data()
        
        # Set resonance array for each time bulk
        init_bulk_func_params, init_bulk_err_params = self.__Set_Initial_Resonance_Arrays()

        #Process all bulks
        print('Begin diurnal data processing ..')
        for j in range (0,self.__number_of_bulks):
            self.__bulks[j], self.__bulk_func_params[j], self.__bulk_err_params[j] = self.__Calculate_FFT_and_Resonances(data[j], init_bulk_func_params[j], init_bulk_err_params[j])
            # Check resonance paramters
            self.__bulk_func_params[j], self.__bulk_err_params[j] = self.__Check_Resonance_Parameters_Adequacy(self.__bulk_func_params[j], self.__bulk_err_params[j])
        print('Diurnal data processing finished.')
        
        # Set frequency array for output spectrum
        self.__freq_arr = np.linspace(self.__h5_low_freq, self.__h5_up_freq, self.__h5_freq_up_index-self.__h5_freq_low_index)

        if (self.__calibration_file==''):
            print("There is no calibration file. All resonance amplitudes are in terms of mV/sqrt(Hz)")

    # Processing for a day file
    def Process_h5_day_data(self, name, month_name=''):
        # Set month average SR file
        self.__month_file_path = month_name
        # Calculate parameters of windows and intervals
        self.__Calculate_Mesh_Values()
        
        # Check window parameters
        self.__Check_Window_Sampling()
        
        # Load h5 dictionary with field data
        dict_h5_data = self.__Load_h5_File(name)
        for its in (list(dict_h5_data.keys())):
            #Search number of time_bulk intervals in file
            self.__number_of_bulks = int(len(dict_h5_data[its][:])//self.__number_of_samples_in_bulk)
            data = dict_h5_data[its][0:int(self.__number_of_bulks*self.__number_of_samples_in_bulk)].reshape((self.__number_of_bulks, self.__number_of_samples_in_bulk))
            print("Number_of_bulks_in_component", its, "=", self.__number_of_bulks)
            self.__Initialize_Arrays_For_FFT_data()
            
            # Set resonance array for each time bulk
            init_bulk_func_params, init_bulk_err_params = self.__Set_Initial_Resonance_Arrays()

            #Process all bulks
            print('Begin diurnal data processing for', its, '..')
            for j in range (0,self.__number_of_bulks):
                self.__bulks[j], self.__bulk_func_params[j], self.__bulk_err_params[j] = self.__Calculate_FFT_and_Resonances(data[j], init_bulk_func_params[j], init_bulk_err_params[j])
                # Check resonance paramters
                self.__bulk_func_params[j], self.__bulk_err_params[j] = self.__Check_Resonance_Parameters_Adequacy(self.__bulk_func_params[j], self.__bulk_err_params[j])
            print('Diurnal data processing finished.')
            
            # Set frequency array for output spectrum
            self.__freq_arr = np.linspace(self.__h5_low_freq, self.__h5_up_freq, self.__h5_freq_up_index-self.__h5_freq_low_index)

            if (self.__calibration_file==''):
                print("There is no calibration file. All resonance amplitudes are in terms of mV/sqrt(Hz)")
            self.Make_H5_Day_Dump(its+'.h5')


        
    def __Calculate_FFT_and_Resonances(self, data, init_bulk_func_params, init_bulk_err_params):
        # Calculate spectrum and resonance parameters for a bulk
        bulk = data
        sum_fourier_for_bulk = np.empty([1,1])
        actual_number_of_windows = 0
        for i in range (0,self.__number_of_windows_in_bulk):
            window_begin_point_in_bulk = i*self.__number_of_samples_in_window//2
            current_window = bulk[window_begin_point_in_bulk:window_begin_point_in_bulk+self.__number_of_samples_in_window]
            if (max(np.abs(current_window))<self.__signal_threshold):
                # Fourier transform for window smoothed function. Window smoothed function is a production of hunn function and data.
                modified_window = self.__Modify_window_for_fft(current_window)

                # Calculate fourier window
                fourier_for_window = fft(modified_window)[0:len(modified_window)//2]

                
                #Adjust amplitude spectrum.
                #   1. Take square module of fourier_window.
                #   2. Divide each value by the product of sampling frequency and number of samples in a window
                adjusted_fourier_for_window = self.__Adjust_fourier_window(fourier_for_window)
                
                #Sum window spectrum to a bulk spectrum
                if (i==0):
                    sum_fourier_for_bulk = np.abs(adjusted_fourier_for_window)
                else:
                    sum_fourier_for_bulk += np.abs(adjusted_fourier_for_window)
                actual_number_of_windows += 1
        # Average adjusted square module through all windows in a bulk and take a square root
        fourier_for_bulk = np.sqrt(sum_fourier_for_bulk/actual_number_of_windows)
        
        # Calibrate spectrum if there is calibration function
        fourier_for_bulk = self.__Calibrate_spectrum(fourier_for_bulk)
        
        # Create fitting curve
        approx_params, errs = self.__Create_curve_fitting_fuction(fourier_for_bulk**2, init_bulk_func_params)

        # Output fourier array is chopped for output
        return fourier_for_bulk[self.__h5_freq_low_index:self.__h5_freq_up_index], approx_params, errs
            
    def __Check_Resonance_Parameters_Adequacy(self,func_params, err_params):
        # Check fit function for accuracy
        # 1. Check curl parameters
        if not(6.0<func_params[3]<8.5 and 0.0<err_params[3]<0.15 and 0.0<func_params[0] and 0.0<func_params[6]):
            for comp in (0,3,6):
                func_params[comp] = float('nan')
                err_params[comp] = float('nan')

        #Check second resonance
        if not(13.0<func_params[4]<15.0 and 0.0<err_params[4]<0.15 and 0.0<func_params[1] and 0.0<func_params[7]):
            for comp in (1,4,7):
                func_params[comp] = float('nan')
                err_params[comp] = float('nan')

        #Check third resonance
        if not(18.0<func_params[5]<22.0 and 0.0<err_params[5]<0.2 and 0.0<func_params[2] and 0.0<func_params[8]):
            for comp in (2,5,8):
                func_params[comp] = float('nan')
                err_params[comp] = float('nan')
        return func_params, err_params
    
    
    def __Calibrate_spectrum(self, fourier):
        try:
            if (self.__calibration_file!=''):
                # Load calibration file
                norm_dat = np.loadtxt('norm_function_chopper.dat')
                freq_dat = norm_dat.transpose()[0][10:40]
                coeff_dat = norm_dat.transpose()[1][10:40]
                fitted_data = np.polyfit(freq_dat, coeff_dat, 14)
                p = np.poly1d(fitted_data)
                
                for i in range (1, fourier.size):
                    fourier[i] = fourier[i]/1000   # Convert from mV into V
                    fourier[i] = fourier[i]/(p(i/self.__window_width)*(i/self.__window_width))    # Multiply with calibration function V/sqrt(Hz) / (V/nT/Hz * Hz)
                    fourier[i] = fourier[i]*1000    # Convert into pT
            return fourier
        except OSError:
            self.__calibration_file = ''
            print("There is no such calibration file!")
            exit(1)


    def __Adjust_fourier_window(self, fourier_window):
        # 1. Find a square module of the fourier spectrum
        square_module = fourier_window**2
        # 1.1 Incorporate factor 2 in the corresponding negative frequencies
        square_module *= 2;
        # 2. Divide by the product of the sampling frequency and the number of samples
        square_module = square_module/(self.__sampling_frequency*self.__number_of_samples_in_window)
        return square_module

  #  def __Add_nulls_to_window(self, window):
  #      #For each window calculate FFT, so a window should have a power of 2 samples
  #      new_number_of_samples = self.__fft_window_sampling
  #      diff = new_number_of_samples - len(window)
  #      return np.concatenate( (np.zeros(diff//2), window, np.zeros(diff//2 + diff%2)) )

    def __Calculate_Hann_function_array(self):
        hann = np.zeros(self.__number_of_samples_in_window)
        for i in range (0,self.__number_of_samples_in_window):
            hann[i] = 1.0 - (math.cos((i-1)*math.pi/self.__number_of_samples_in_window))**2
        
        hann /= math.sqrt(np.sum(hann**2)/self.__number_of_samples_in_window)
        return hann
        
    def __Modify_window_for_fft(self, window):
     #   zero_padded_window = self.__Add_nulls_to_window(window)
        #For a window calculate array according to Hann window function
        hunn_function = self.__Calculate_Hann_function_array()
        #return modified data for a window
        return window*hunn_function
        
        
    # Set initial array of resonance parameters for each time bulk: from average month file or standard array
    def __Set_Initial_Resonance_Arrays(self):
        self.__bulk_func_params = np.zeros((self.__number_of_bulks, 11))
        self.__bulk_err_params = np.zeros((self.__number_of_bulks, 11))
        init_bulk_func_params = self.__bulk_func_params
        init_bulk_err_params = self.__bulk_err_params
        
        # Try to load mena month file
        if (self.__month_file_path!=''):
            h = h5py.File(self.__month_file_path, 'r')
            init_bulk_func_params = np.array(h['approximation curve parameters'])
            init_bulk_err_params = np.array(h['approximation curve errors'])
            for i in range(len(init_bulk_func_params)):
                if (np.sum(np.isnan(init_bulk_func_params[i]))>0):
                    init_bulk_func_params[i] = self.__standart_resonance_parameters
            h.close()
            print("Init average month parameters uploaded ...")
        else:
            print("Init average month parameters are not found .. standart average parameters will be used ..")
            for i in range (0, self.__number_of_bulks):
                init_bulk_func_params[i] = self.__standart_resonance_parameters
        return init_bulk_func_params, init_bulk_err_params

    def __Initialize_Arrays_For_FFT_data(self):
        #init array for bulk data
        self.__h5_freq_low_index = int(math.floor(self.__h5_low_freq*self.__window_width))
        self.__h5_freq_up_index = int(math.floor(self.__h5_up_freq*self.__window_width))
        self.__bulks = np.empty( (self.__number_of_bulks, self.__h5_freq_up_index-self.__h5_freq_low_index) )
        self.__freq_arr = np.empty(self.__h5_freq_up_index-self.__h5_freq_low_index)
        
    # Calculate initial values
    def __Calculate_Mesh_Values(self):
        print('Processing parameters (if needed, they can be changed) ..')
        # Calculate initial values (do not change order!)
        dt = 1.0/self.__sampling_frequency  #time discretization
        self.__number_of_samples_in_window = int(math.floor(self.__window_width/dt))//2*2 # number of samples in a window should be even
        print("Number_of_samples_in_window = ", self.__number_of_samples_in_window)
        
        # number of time windows in one bulk (multiplier 2 is because windows overlap each other of a half of the length)
        self.__number_of_windows_in_bulk = int(math.floor(self.__time_bulk/dt)*2)//(self.__number_of_samples_in_window) - 1
        print("Number_of_windows_in_bulk = ", self.__number_of_windows_in_bulk)
        
        self.__number_of_samples_in_bulk = (self.__number_of_windows_in_bulk+1)*(self.__number_of_samples_in_window//2)  #number of points in one time bulk
        print("Number_of_samples_in_bulk = ", self.__number_of_samples_in_bulk)

    # Checks if current window length in terms of seconds fits into array of window samples window_sampling
    def __Check_Window_Sampling(self):
        if (self.__window_width*self.__sampling_frequency==self.__fft_window_sampling):
            print("Window lenght for fft is good ..")
            print("_______________\n")
        elif (self.__window_width*self.__sampling_frequency<self.__fft_window_sampling):
            print("Window sampling is larger than data. Zero padding technique is needed")
        else:
            print("Window length for fft is too small. Try to set larger 'window_width' parameter!")
            exit(1)
        
    # Load data from input file
    def __Load_txt_File(self, file_name):
        try:
            print("Uploading ", file_name, " ...")
            data = np.loadtxt(file_name, dtype='double')
        except OSError:
            print("There is no input file. Check the file name...")
        #Search number of time_bulk intervals in file
        self.__number_of_bulks = int(len(data)//self.__number_of_samples_in_bulk)
        print("Number_of_bulks_in_file =", self.__number_of_bulks)
        return data[0:int(self.__number_of_bulks*self.__number_of_samples_in_bulk)].reshape((self.__number_of_bulks, self.__number_of_samples_in_bulk))
    
    def __Load_h5_File(self, file_name):
        try:
            print("Uploading ", file_name, " ...")
            h = h5py.File(file_name, 'r')
            return h
        except OSError:
            print("There is no input file. Check the file name...")
        

    def Make_H5_Day_Dump(self, data_name, h5_spectrum_flag=1):
        #data_path = "./anual_average/" + str(year) + "/" + str(month) + "/" + component + "_" + str(day) + '.h5'
        hf = h5py.File(data_name, 'w')
        if (h5_spectrum_flag == 1):
            dat = hf.create_dataset('fourier spectrum', data=self.__bulks)
            dat.attrs['accumulation interval [min]'] = self.__time_bulk/60
            dat.attrs['min frequency [Hz]'] = self.__h5_low_freq
            dat.attrs['max frequency [Hz]'] = self.__h5_up_freq
            dat.attrs['delta frequency [Hz]'] = self.__freq_arr[2]-self.__freq_arr[1]
            dat.attrs['window width [s]'] = self.__window_width
            if (self.__calibration_file==''):
                dat.attrs['Amplitude'] = 'mV/sqrt(Hz)'
            else:
                dat.attrs['Amplitude'] = 'pT/sqrt(Hz)'

        dat = hf.create_dataset('approximation curve parameters', data=self.__bulk_func_params)
        dat.attrs['function type'] = 'Lorenzian'
        dat.attrs['min frequency [Hz]'] = 6
        dat.attrs['max frequency [Hz]'] = 24
        dat = hf.create_dataset('approximation curve errors', data=self.__bulk_err_params)
        hf.close()
        print('Binary output file has been made.')

    def __Create_curve_fitting_fuction(self,spectrum, initial_parameters):
        # Detect frequency step for spectrum
        delta_f = self.__sampling_frequency/self.__fft_window_sampling
        
        # Construct spectrum axis
        freq_axis = np.arange(0, delta_f*len(spectrum), delta_f)
        
        # Calculate indexes for freq_low_boundary and freq_high_boundary
        freq_low_index = int(math.floor(self.__f_min/delta_f))
        freq_high_index = int(math.floor(self.__f_max/delta_f))
        # Cut spectrum to needed value
        cutted_spectrum = spectrum[freq_low_index:freq_high_index]
        cutted_freq_axis = freq_axis[freq_low_index:freq_high_index]
        
        # Adjust initial amplitudes according to spectrum and initial frequencies
        adjusted_params = initial_parameters
        for i in range(0,3):
            fp = int(math.floor((adjusted_params[i+3]-self.__f_min)/delta_f))
            if fp<0: 
                fp=0
            elif fp>=len(cutted_freq_axis):
                fp = len(cutted_freq_axis)-1
            adjusted_params[i] = cutted_spectrum[fp]

        params = Parameters()
        self.__define_minimization_parameters(params, adjusted_params)
        out_s0 = minimize(self.__minimize_residual_function, params, args=(cutted_freq_axis, cutted_spectrum), method = 'leastsq')
        return self.__get_parameters(out_s0)

    def __get_parameters(self,output):
        val= np.zeros(9)
        err= np.zeros(9)
        val[0]=output.params['a1'].value
        val[1]=output.params['a2'].value
        val[2]=output.params['a3'].value
        val[3]=output.params['f1'].value
        val[4]=output.params['f2'].value
        val[5]=output.params['f3'].value    
        val[6]=output.params['s1'].value
        val[7]=output.params['s2'].value
        val[8]=output.params['s3'].value


        err[0]=output.params['a1'].stderr
        err[1]=output.params['a2'].stderr
        err[2]=output.params['a3'].stderr
        err[3]=output.params['f1'].stderr
        err[4]=output.params['f2'].stderr
        err[5]=output.params['f3'].stderr   
        err[6]=output.params['s1'].stderr
        err[7]=output.params['s2'].stderr
        err[8]=output.params['s3'].stderr
        return val,err

    def __minimize_residual_function(self, params, x, data):
        v= params.valuesdict()     
        model = v['a1']/(1+((x-v['f1'])**2)/v['s1']**2)+\
                    v['a2']/(1+((x-v['f2'])**2)/v['s2']**2)+\
                    v['a3']/(1+((x-v['f3'])**2)/v['s3']**2)    
        return (data-model)

    def __define_minimization_parameters(self,params,val):
        params.add('a1', value = val[0])
        params.add('a2', value = val[1])
        params.add('a3', value = val[2])    
        params.add('f1', value = val[3])
        params.add('f2', value = val[4])
        params.add('f3', value = val[5])    
        params.add('s1', value= val[6])
        params.add('s2', value= val[7])
        params.add('s3', value= val[8])    

    
    def Make_TXT_Day_Dump(self,day, month, year, bulk, func_params, err_params):
        param_path = "./anual_average/" + str(year) + "/" + str(month) + "/parameter_" + component + "_" + str(day)
        error_path = "./anual_average/" + str(year) + "/" + str(month) + "/error_" + component + "_" + str(day)
        data_path = "./anual_average/" + str(year) + "/" + str(month) + "/data_" + component + "_" + str(day)
        np.savetxt(param_path, func_params)
        print("Data for the ", day, "-", month, "-", year, "saved into ", param_path, ".")
        np.savetxt(error_path, err_params)
        print("Data for the ", day, "-", month, "-", year, "saved into ", error_path, ".")
    #  np.savetxt(data_path, bulk)
    
