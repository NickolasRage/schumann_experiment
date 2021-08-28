import schumann_processing_class as sch

p = sch.Schumann_processing()
#p.Set_Calibration_File('norm_function_chopper.dat')

# Import binary h5 file
p.Process_h5_day_data('../../final/data.h5')

#p.Process_text_day_data('../../data/2019/2/052_V01_C02_R000_THx_BL_256_2.dat')
#p.Make_H5_Day_Dump('test.h5')
