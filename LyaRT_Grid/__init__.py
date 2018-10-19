import os
import os.path

#import imp

import sys
import shutil

import urllib

import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

import pickle

#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Check_if_DATA_files_are_found():

    this_dir, this_filename = os.path.split(__file__)

    Bool_1 = True
     
    arxiv_with_file_names = this_dir + '/DATA/List_of_DATA_files'

    with open( arxiv_with_file_names ) as fd:

        for line in fd.readlines():

            arxiv_name = line.strip('\n')

            Bool_1 = Bool_1 * os.path.isfile( this_dir + '/DATA/' + arxiv_name )

    return Bool_1
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Download_data():

    this_dir, this_filename = os.path.split(__file__)

    arxiv_with_file_names = this_dir + '/DATA/List_of_DATA_files'

    file_where_to_store_data = this_dir + '/DATA/'

    print( 'This package is stored in ', this_dir , '(Please, note that we are not spying you.)' )

    http_url = 'http://www.cefca.es/people/~sidgurung/ShouT/ShouT/DATA/'


    testfile = urllib.URLopener()

    with open( arxiv_with_file_names ) as fd:

        for line in fd.readlines():

            arxiv_name = line.strip('\n')

            print( 'Downloaing...' , http_url + arxiv_name )

            testfile.retrieve( http_url + arxiv_name , arxiv_name )

            print( '--> Done!' )

            print( 'Moving Downloaded file to' , file_where_to_store_data )

            shutil.move( arxiv_name , file_where_to_store_data + arxiv_name )

            print( '--> Done' )

    if Check_if_DATA_files_are_found():
        print( '\nHey man, looks like everything is done! That is brilliant!' )

    else:
        print( 'This is weird... We just downloaded everthing but the files are not found...Exiting...')
        print( 'Error. Human is dead. Mismatch.')
        sys.exit()

    return
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def load_machine_fesc( Machine , property_name , Geometry , INSIDE_BICONE=True ):

    '''
        This functions gives you the trained model that you want to use.
    '''

    Machine_Set = [ 'KN' , 'Grad' , 'Tree' , 'Forest'  ]

    Geometry_Set = [ 'Thin_Shell'  , 'Galactic_Wind'  , 'Bicone_X_Slab' ]

    dirs_Set     = [ 'Thin_Shell/' , 'Galactic_Wind/' , 'Bicone_X_Slab/' ]

    geo_code     = [ 'thin'        , 'wind'           , 'Bicone_X_Slab' ]

    Property_Set = [ 'KKK' , 'CCC' , 'LLL' , 'f_esc' ]

    assert property_name in Property_Set , "Houston we've got a problem, Error Code = 23452345.7523"

    index = np.where( Geometry == np.array(Geometry_Set) )[0][0]

    this_dir, this_filename = os.path.split(__file__)

    filename_root = 'DATA/finalized_model_'+ geo_code[index] +'_f_esc_' + Machine + '_' + property_name 

    if Geometry == 'Bicone_X_Slab':
        filename_root += '_Inside_Bicone_' + str(INSIDE_BICONE)

    filename = filename_root + '.sav'

    filename = os.path.join(this_dir, filename)

    loaded_model = pickle.load(open(filename, 'rb'))

    return  loaded_model
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Analytic_f_esc_Thin_Shell( V_Arr , logNH_Arr , ta_Arr ):

    NH18 = 10 ** ( logNH_Arr - 18 ) 

    # Old MCMC
    #c11 = 10**(2.109185)
    #c12 = -10**(2.745113)
    #c13 = 10**(2.954875)
    #c21 = 10**(-1.785661)
    #c22 = -10**(-0.7302781)
    #c23 = 10**(-0.1554347)
    #c24 = -10**(0.1517145)
    #c3 = 10**(-0.03688789)
    #c4 = 10**(-1.556422)


    #New MCMC
    c11 = 10**(1.90526)
    c12 = -10**(2.0399)
    c13 = 10**(2.34829)
    c21 = 10**(-3.138837)
    c22 = -10**(-1.92151)
    c23 = 10**(-1.1860205000000001)
    c24 = -10**(-0.1480042)
    c3 = 10**(0.0530715)
    c4 = 10**(-2.743455)

    C1 = ( ( np.log10(NH18) ) ** 2 ) * c11 + np.log10(NH18) * c12 + c13
    y  = np.log10(NH18)
    C2 = c21*y*y*y + c22*y*y + c23*y + c24
    C3 = c3
    C4 = c4

    K1 = C1 * ( V_Arr ** C2 )
    K2 = C3 * ( V_Arr ** C4 )

    fesc =  1. / np.cosh( np.sqrt( K1 * ( ta_Arr ** K2 ) ) )

    return fesc
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Analytic_f_esc_Wind( V_Arr , logNH_Arr , ta_Arr ):

    NH18 = 10 ** ( logNH_Arr - 18 )


    # New MCMC
    c11 = 10**(0.4852541)
    c12 = 10**(-0.2006394)
    c21 = 10**(-1.912059)
    c22 = -10**(-0.6380347)
    c3 = 10**(0.046314074999999996)
    c4 = 10**(-1.782037)

    C1  = c11 * ( NH18 ** c12 )
    C2  = c21 * np.log10( NH18 )**2 + c22 * np.log10(NH18) #+ c23
    C3  = c3
    C4  = c4

    K1  = C1 * V_Arr ** C2
    K2  = C3 * V_Arr ** C4

    fesc = 1./ np.cosh( np.sqrt( K1 * ta_Arr ** K2 ) )

    return fesc
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Analytic( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr=None , Machine_Learning_Algorithm=None ):

    Geometry_Set = [ 'Thin_Shell'  , 'Galactic_Wind'  ]

    assert Geometry in Geometry_Set , 'The geometry ' + Geometry + ' is nor supported in MODE=Analytic , only Thin_Shell and Galactic_Wind'

    logNH_Arr         = np.atleast_1d( logNH_Arr )
    ta_Arr            = np.atleast_1d(    ta_Arr )
    V_Arr             = np.atleast_1d(     V_Arr )

    if Geometry == 'Thin_Shell' :
        f_esc_Arr = Analytic_f_esc_Thin_Shell( V_Arr , logNH_Arr , ta_Arr )
    if Geometry == 'Galactic_Wind' :
        f_esc_Arr = Analytic_f_esc_Wind( V_Arr , logNH_Arr , ta_Arr )

    return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def fesc_of_ta_Thin_and_Wind( ta , CCC , KKK ):

    f_esc = 1./np.cosh( np.sqrt( CCC * (ta**KKK) ) )

    return f_esc
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def fesc_of_ta_Bicone( ta , CCC , KKK , LLL ):
 
    f_esc = LLL * 1./np.cosh( np.sqrt( CCC * (ta**KKK) ) )
 
    return f_esc
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Machine_Parameter( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr=None , Machine_Learning_Algorithm='Tree' ):

        logNH_Arr         = np.atleast_1d( logNH_Arr )
        ta_Arr            = np.atleast_1d(    ta_Arr )
        V_Arr             = np.atleast_1d(     V_Arr )


        if Geometry in [ 'Thin_Shell'  , 'Galactic_Wind'  ] :

            Coor_matrix = np.zeros( len(V_Arr) * 2 ).reshape( len(V_Arr) , 2 )

            Coor_matrix[ : , 0 ] = V_Arr
            Coor_matrix[ : , 1 ] = logNH_Arr

            CCC_machine = load_machine_fesc( Machine_Learning_Algorithm , 'CCC' , Geometry )
            KKK_machine = load_machine_fesc( Machine_Learning_Algorithm , 'KKK' , Geometry )

            CCC_model_Arr  = CCC_machine.predict( Coor_matrix )
            KKK_model_Arr  = KKK_machine.predict( Coor_matrix )

            f_esc_Arr = fesc_of_ta_Thin_and_Wind( ta_Arr , CCC_model_Arr , KKK_model_Arr )


        if Geometry in [ 'Bicone_X_Slab' ] :

            assert not Inside_Bicone_Arr is None , 'Inside_Bicone_Arr give is None or none Inside_Bicone_Arr was given. If the geometry is Bicone_X_Slab it is necesary to give a Inside_Bicone_Arr'

            Inside_Bicone_Arr = np.atleast_1d( Inside_Bicone_Arr )

            f_esc_Arr = np.zeros( len(V_Arr) )

            ##################

            if sum( Inside_Bicone_Arr ) > 0 :

                Coor_matrix = np.zeros( sum( Inside_Bicone_Arr ) * 2 ).reshape( sum( Inside_Bicone_Arr ) , 2 )

                Coor_matrix[ : , 0 ] = V_Arr[     Inside_Bicone_Arr ]
                Coor_matrix[ : , 1 ] = logNH_Arr[ Inside_Bicone_Arr ]

                CCC_machine_in = load_machine_fesc( Machine_Learning_Algorithm , 'CCC' , Geometry , INSIDE_BICONE=True )
                KKK_machine_in = load_machine_fesc( Machine_Learning_Algorithm , 'KKK' , Geometry , INSIDE_BICONE=True )
                LLL_machine_in = load_machine_fesc( Machine_Learning_Algorithm , 'LLL' , Geometry , INSIDE_BICONE=True )

                CCC_model_in_Arr  = CCC_machine_in.predict( Coor_matrix )
                KKK_model_in_Arr  = KKK_machine_in.predict( Coor_matrix )
                LLL_model_in_Arr  = LLL_machine_in.predict( Coor_matrix )

                f_esc_Arr[ Inside_Bicone_Arr ] = fesc_of_ta_Bicone( ta_Arr[ Inside_Bicone_Arr ] , CCC_model_in_Arr , KKK_model_in_Arr , LLL_model_in_Arr )

            ##################

            if sum( ~Inside_Bicone_Arr ) > 0 :
                Coor_matrix = np.zeros( sum( ~Inside_Bicone_Arr ) * 2 ).reshape( sum( ~Inside_Bicone_Arr ) , 2 )

                Coor_matrix[ : , 0 ] = V_Arr[     ~Inside_Bicone_Arr ]
                Coor_matrix[ : , 1 ] = logNH_Arr[ ~Inside_Bicone_Arr ]

                CCC_machine_out = load_machine_fesc( Machine_Learning_Algorithm , 'CCC' , Geometry , INSIDE_BICONE=False )
                KKK_machine_out = load_machine_fesc( Machine_Learning_Algorithm , 'KKK' , Geometry , INSIDE_BICONE=False )
                LLL_machine_out = load_machine_fesc( Machine_Learning_Algorithm , 'LLL' , Geometry , INSIDE_BICONE=False )

                CCC_model_out_Arr  = CCC_machine_out.predict( Coor_matrix )
                KKK_model_out_Arr  = KKK_machine_out.predict( Coor_matrix )
                LLL_model_out_Arr  = LLL_machine_out.predict( Coor_matrix )

                f_esc_Arr[ ~Inside_Bicone_Arr ] = fesc_of_ta_Bicone( ta_Arr[ ~Inside_Bicone_Arr ] , CCC_model_out_Arr , KKK_model_out_Arr , LLL_model_out_Arr )


        return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Machine_Values( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr=None , Machine_Learning_Algorithm='Tree' ):

        logNH_Arr         = np.atleast_1d( logNH_Arr )
        ta_Arr            = np.atleast_1d(    ta_Arr )
        V_Arr             = np.atleast_1d(     V_Arr )

        if Geometry in [ 'Thin_Shell'  , 'Galactic_Wind'  ] :

            loaded_model = load_machine_fesc( Machine_Learning_Algorithm , 'f_esc' , Geometry )

            Coor_matrix = np.zeros( len(V_Arr) * 3 ).reshape( len(V_Arr) , 3 )

            Coor_matrix[ : , 0 ] = V_Arr
            Coor_matrix[ : , 1 ] = logNH_Arr
            Coor_matrix[ : , 2 ] = np.log10(ta_Arr)
       
            f_esc_Arr = loaded_model.predict( Coor_matrix )

        if Geometry in [ 'Bicone_X_Slab' ] :

            Inside_Bicone_Arr = np.atleast_1d( Inside_Bicone_Arr )

            f_esc_Arr = np.zeros( len(V_Arr) )

            ##################

            if sum( Inside_Bicone_Arr ) > 0 :

                loaded_model_inside = load_machine_fesc( Machine_Learning_Algorithm , 'f_esc' , Geometry , INSIDE_BICONE=True )

                Coor_matrix = np.zeros( sum( Inside_Bicone_Arr ) * 3 ).reshape( sum( Inside_Bicone_Arr ) , 3 )
    
                Coor_matrix[ : , 0 ] = V_Arr[            Inside_Bicone_Arr ]
                Coor_matrix[ : , 1 ] = logNH_Arr[        Inside_Bicone_Arr ]
                Coor_matrix[ : , 2 ] = np.log10(ta_Arr)[ Inside_Bicone_Arr ]
    
                f_esc_Arr[ Inside_Bicone_Arr ] = loaded_model_inside.predict( Coor_matrix )
    
            ##################
    
            if sum( ~Inside_Bicone_Arr ) > 0 :

                loaded_model_outside = load_machine_fesc( Machine_Learning_Algorithm , 'f_esc' , Geometry , INSIDE_BICONE=False )
    
                Coor_matrix = np.zeros( sum( ~Inside_Bicone_Arr ) * 3 ).reshape( sum( ~Inside_Bicone_Arr ) , 3 )
    
                Coor_matrix[ : , 0 ] = V_Arr[            ~Inside_Bicone_Arr ]
                Coor_matrix[ : , 1 ] = logNH_Arr[        ~Inside_Bicone_Arr ]
                Coor_matrix[ : , 2 ] = np.log10(ta_Arr)[ ~Inside_Bicone_Arr ]
    
                f_esc_Arr[ ~Inside_Bicone_Arr ] = loaded_model_outside.predict( Coor_matrix )

        return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Linear_2D_interpolator( X_prob , Y_prob , X_grid , Y_grid , Field_in_grid ):

    INDEX_X = np.where( ( X_grid < X_prob ) )[0][-1]
    INDEX_Y = np.where( ( Y_grid < Y_prob ) )[0][-1]

    dX_grid = X_grid[ INDEX_X + 1 ] - X_grid[ INDEX_X ]
    dY_grid = Y_grid[ INDEX_Y + 1 ] - Y_grid[ INDEX_Y ]

    X_min_grid = X_grid[ INDEX_X     ]
    Y_min_grid = Y_grid[ INDEX_Y     ]

    Xprob_X0 = ( X_prob - X_min_grid ) * 1. / dX_grid
    Yprob_Y0 = ( Y_prob - Y_min_grid ) * 1. / dY_grid

    Area1 = ( 1. - Xprob_X0 ) * ( 1. - Yprob_Y0 )
    Area2 = ( 1. - Xprob_X0 ) * (      Yprob_Y0 )
    Area3 = (      Xprob_X0 ) * (      Yprob_Y0 )
    Area4 = (      Xprob_X0 ) * ( 1. - Yprob_Y0 )

    Field1 = Field_in_grid[ INDEX_X     , INDEX_Y     ]
    Field2 = Field_in_grid[ INDEX_X     , INDEX_Y + 1 ]
    Field3 = Field_in_grid[ INDEX_X + 1 , INDEX_Y + 1 ]
    Field4 = Field_in_grid[ INDEX_X + 1 , INDEX_Y     ]

    Field_at_the_prob_point = Area1 * Field1 + Area2 * Field2 + Area3 * Field3 + Area4 * Field4

    return Field_at_the_prob_point
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Linear_3D_interpolator( X_prob , Y_prob , Z_prob , X_grid , Y_grid , Z_grid , Field_in_grid ):

    INDEX_X = np.where( ( X_grid < X_prob ) )[0][-1]
    INDEX_Y = np.where( ( Y_grid < Y_prob ) )[0][-1]
    INDEX_Z = np.where( ( Z_grid < Z_prob ) )[0][-1]
   

    dX_grid = X_grid[ INDEX_X + 1 ] - X_grid[ INDEX_X ]
    dY_grid = Y_grid[ INDEX_Y + 1 ] - Y_grid[ INDEX_Y ]
    dZ_grid = Z_grid[ INDEX_Z + 1 ] - Z_grid[ INDEX_Z ]

    X_min_grid = X_grid[ INDEX_X     ]
    Y_min_grid = Y_grid[ INDEX_Y     ]
    Z_min_grid = Z_grid[ INDEX_Z     ]

    Xprob_X0 = ( X_prob - X_min_grid ) * 1. / dX_grid
    Yprob_Y0 = ( Y_prob - Y_min_grid ) * 1. / dY_grid
    Zprob_Z0 = ( Z_prob - Z_min_grid ) * 1. / dZ_grid

    Vol1 = ( 1. - Xprob_X0 ) * ( 1. - Yprob_Y0 ) * ( 1. - Zprob_Z0 )
    Vol2 = ( 1. - Xprob_X0 ) * (      Yprob_Y0 ) * ( 1. - Zprob_Z0 )
    Vol3 = ( 1. - Xprob_X0 ) * (      Yprob_Y0 ) * (      Zprob_Z0 )
    Vol4 = ( 1. - Xprob_X0 ) * ( 1. - Yprob_Y0 ) * (      Zprob_Z0 )

    Vol5 = (      Xprob_X0 ) * ( 1. - Yprob_Y0 ) * ( 1. - Zprob_Z0 )
    Vol6 = (      Xprob_X0 ) * (      Yprob_Y0 ) * ( 1. - Zprob_Z0 )
    Vol7 = (      Xprob_X0 ) * (      Yprob_Y0 ) * (      Zprob_Z0 )
    Vol8 = (      Xprob_X0 ) * ( 1. - Yprob_Y0 ) * (      Zprob_Z0 )

    Field1 = Field_in_grid[ INDEX_X     , INDEX_Y     , INDEX_Z     ]
    Field2 = Field_in_grid[ INDEX_X     , INDEX_Y + 1 , INDEX_Z     ]
    Field3 = Field_in_grid[ INDEX_X     , INDEX_Y + 1 , INDEX_Z + 1 ]
    Field4 = Field_in_grid[ INDEX_X     , INDEX_Y     , INDEX_Z + 1 ]
    Field5 = Field_in_grid[ INDEX_X + 1 , INDEX_Y     , INDEX_Z     ]
    Field6 = Field_in_grid[ INDEX_X + 1 , INDEX_Y + 1 , INDEX_Z     ]
    Field7 = Field_in_grid[ INDEX_X + 1 , INDEX_Y + 1 , INDEX_Z + 1 ]
    Field8 = Field_in_grid[ INDEX_X + 1 , INDEX_Y     , INDEX_Z + 1 ]

    Field_at_the_prob_point = Vol1 * Field1 + Vol2 * Field2 + Vol3 * Field3 + Vol4 * Field4 + Vol5 * Field5 + Vol6 * Field6 + Vol7 * Field7 + Vol8 * Field8

    return Field_at_the_prob_point
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def load_Grid_fesc( Geometry , MODE , INSIDE_BICONE=True ):

    Geometry_Set = [ 'Thin_Shell'  , 'Galactic_Wind'  , 'Bicone_X_Slab' ]

    dirs_Set     = [ 'Thin_Shell/' , 'Galactic_Wind/' , 'Bicone_X_Slab/' ]

    geo_code     = [ 'Thin_Shell'  , 'Wind'           , 'Bicone_X_Slab' ]

    MODE_Set = [ 'Parameters' , 'values' ]

    index = np.where( Geometry == np.array(Geometry_Set) )[0][0]

    filename_root = 'DATA/Dictonary_'+ geo_code[index] +'_Grid_f_esc_' + MODE 

    if Geometry == 'Bicone_X_Slab':
        filename_root += '_Inside_Bicone_' + str(INSIDE_BICONE)

    filename = filename_root + '.npy'

    this_dir, this_filename = os.path.split(__file__)

    filename = os.path.join(this_dir, filename)

    loaded_model = np.load( filename ).item()

    return loaded_model
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_f_esc_Arrays_2D_grid( V_Arr , logNH_Arr , ta_Arr , Grid_Dictionary , Geometry ):

    V_Arr_Grid     = Grid_Dictionary[     'V_Arr' ]

    logNH_Arr_Grid = Grid_Dictionary[ 'logNH_Arr' ]

    logta_Arr_Grid = Grid_Dictionary[ 'logta_Arr' ]

    Grid           = Grid_Dictionary[    'Grid'   ]

    N_objects = len( V_Arr )

    CCC_Arr_evaluated = np.zeros( N_objects )
    KKK_Arr_evaluated = np.zeros( N_objects )

    ###################

    if Geometry in [ 'Thin_Shell'  , 'Galactic_Wind'  ] :

        for INDEX in range( 0 , N_objects ):

            CCC_Arr_evaluated[ INDEX ] , KKK_Arr_evaluated[ INDEX ] = Linear_2D_interpolator( V_Arr[INDEX] , logNH_Arr[INDEX] , V_Arr_Grid , logNH_Arr_Grid , Grid )

        f_esc_Arr = fesc_of_ta_Thin_and_Wind( ta_Arr , CCC_Arr_evaluated , KKK_Arr_evaluated )

    ###################

    if Geometry in [ 'Bicone_X_Slab' ] :
    
        LLL_Arr_evaluated = np.zeros( N_objects )

        for INDEX in range( 0 , N_objects ):

            CCC_Arr_evaluated[ INDEX ] , KKK_Arr_evaluated[ INDEX ] , LLL_Arr_evaluated[ INDEX ] = Linear_2D_interpolator( V_Arr[INDEX] , logNH_Arr[INDEX] , V_Arr_Grid , logNH_Arr_Grid , Grid )

        f_esc_Arr = fesc_of_ta_Bicone( ta_Arr , CCC_Arr_evaluated , KKK_Arr_evaluated , LLL_Arr_evaluated )


    return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_fesc_Arrays_3D_grid( V_Arr , logNH_Arr , ta_Arr , Grid_Dictionary ):

    V_Arr_Grid     = Grid_Dictionary[     'V_Arr' ]

    logNH_Arr_Grid = Grid_Dictionary[ 'logNH_Arr' ]

    logta_Arr_Grid = Grid_Dictionary[ 'logta_Arr' ]

    Grid           = Grid_Dictionary[    'Grid'   ]

    logta_Arr = np.log10( ta_Arr )

    N_objects = len( V_Arr )

    f_esc_Arr_evaluated = np.zeros( N_objects )

    for INDEX in range( 0 , N_objects ):

        f_esc_Arr_evaluated[ INDEX ] = Linear_3D_interpolator( V_Arr[INDEX] , logNH_Arr[INDEX] , logta_Arr[INDEX] , V_Arr_Grid , logNH_Arr_Grid , logta_Arr_Grid , Grid )


    return f_esc_Arr_evaluated
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Interpolation_Values( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr=None , Machine_Learning_Algorithm=None ):

    logNH_Arr         = np.atleast_1d( logNH_Arr )
    ta_Arr            = np.atleast_1d(    ta_Arr )
    V_Arr             = np.atleast_1d(     V_Arr )


    if Geometry in [ 'Thin_Shell'  , 'Galactic_Wind'  ] :

        DATA_DICTIONAY = load_Grid_fesc( Geometry , 'values' )

        f_esc_Arr = Interpolate_fesc_Arrays_3D_grid( V_Arr , logNH_Arr , ta_Arr , DATA_DICTIONAY )


    if Geometry in [ 'Bicone_X_Slab' ] and not Inside_Bicone_Arr is None :

        Inside_Bicone_Arr = np.atleast_1d( Inside_Bicone_Arr )

        f_esc_Arr = np.zeros( len(logNH_Arr) )

        ##############

        if sum( Inside_Bicone_Arr ) > 0:
            DATA_DICTIONAY_in = load_Grid_fesc( Geometry , 'values' , INSIDE_BICONE=True )

            f_esc_Arr[ Inside_Bicone_Arr ] = Interpolate_fesc_Arrays_3D_grid( V_Arr[Inside_Bicone_Arr] , logNH_Arr[Inside_Bicone_Arr] , ta_Arr[Inside_Bicone_Arr] , DATA_DICTIONAY_in ) 

        ##############

        if sum( ~Inside_Bicone_Arr ) > 0:
            DATA_DICTIONAY_out = load_Grid_fesc( Geometry , 'values' , INSIDE_BICONE=False )

            f_esc_Arr[ ~Inside_Bicone_Arr ] = Interpolate_fesc_Arrays_3D_grid( V_Arr[~Inside_Bicone_Arr] , logNH_Arr[~Inside_Bicone_Arr] , ta_Arr[~Inside_Bicone_Arr] , DATA_DICTIONAY_out ) 


    return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc_Interpolation_Parameters( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr=None , Machine_Learning_Algorithm=None ):

    logNH_Arr         = np.atleast_1d( logNH_Arr )
    ta_Arr            = np.atleast_1d(    ta_Arr )
    V_Arr             = np.atleast_1d(     V_Arr )


    if Geometry in [ 'Thin_Shell'  , 'Galactic_Wind'  ] :

        DATA_DICTIONAY = load_Grid_fesc( Geometry , 'Parameters' )

        f_esc_Arr = Interpolate_f_esc_Arrays_2D_grid( V_Arr , logNH_Arr , ta_Arr , DATA_DICTIONAY , Geometry )


    if Geometry in [ 'Bicone_X_Slab' ] :

        Inside_Bicone_Arr = np.atleast_1d( Inside_Bicone_Arr )

        f_esc_Arr = np.zeros( len(logNH_Arr) )

        ##############

        DATA_DICTIONAY_in = load_Grid_fesc( Geometry , 'Parameters' , INSIDE_BICONE=True )

        f_esc_Arr[ Inside_Bicone_Arr ] = Interpolate_f_esc_Arrays_2D_grid( V_Arr[Inside_Bicone_Arr] , logNH_Arr[Inside_Bicone_Arr] , ta_Arr[Inside_Bicone_Arr] , DATA_DICTIONAY_in , Geometry ) 

        ##############

        DATA_DICTIONAY_out = load_Grid_fesc( Geometry , 'Parameters' , INSIDE_BICONE=False )

        f_esc_Arr[ ~Inside_Bicone_Arr ] = Interpolate_f_esc_Arrays_2D_grid( V_Arr[~Inside_Bicone_Arr] , logNH_Arr[~Inside_Bicone_Arr] , ta_Arr[~Inside_Bicone_Arr] , DATA_DICTIONAY_out , Geometry ) 


    return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def pre_treatment_f_esc( V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr , MODE ):

    V_Arr     = np.atleast_1d(     V_Arr )
    logNH_Arr = np.atleast_1d( logNH_Arr )
    ta_Arr    = np.atleast_1d(    ta_Arr )

    V_Arr     =     V_Arr.astype(float)
    logNH_Arr = logNH_Arr.astype(float)
    ta_Arr    =    ta_Arr.astype(float)


    bool1 = np.isfinite( V_Arr     )
    bool2 = np.isfinite( logNH_Arr )
    bool3 = np.isfinite( ta_Arr    )

    mask_good = bool1 * bool2 * bool3
    
    assert sum( mask_good ) != 0 , 'All the V-logNH-ta combinations are np.nan, -np.inf or np.inf'

    V_Arr_used     =     V_Arr[ mask_good ]
    logNH_Arr_used = logNH_Arr[ mask_good ]
    ta_Arr_used    =    ta_Arr[ mask_good ]

    #bool4 = ( V_Arr_used <= 100 ) * ( logNH_Arr_used >= 20.5 )
    #V_Arr_used[ bool4 ] = 100.00001

    #============================================#
    bool1 = V_Arr_used < 100.0
    bool2 = logNH_Arr_used >= 20.5

    #aux_V_arr = logNH_Arr_used*-40 + 920.
    aux_V_arr = logNH_Arr_used * ( -100/1.5 ) + ( 100 - ( -100/1.5 * 20.5 ) )

    bool_aux = V_Arr_used > aux_V_arr

    V_Arr_used[     bool1 * bool2 *  bool_aux ] = 100.000001
    logNH_Arr_used[ bool1 * bool2 *~ bool_aux ] = 20.5
    #============================================#

    bool5 = V_Arr_used <= 10.00 
    V_Arr_used[ bool5 ] = 10.000001

    bool6 = V_Arr_used >= 1000
    V_Arr_used[ bool6 ] = 999.9999
    
    bool7 = logNH_Arr_used <= 17.0
    logNH_Arr_used[ bool7 ] = 17.0000001

    bool8 = logNH_Arr_used >= 22.0
    logNH_Arr_used[ bool8 ] = 21.9999

    if MODE=='Raw':
        bool9 = ta_Arr_used <= 10**(-2.5)
        ta_Arr_used[ bool9 ] = 10**(-2.499999)

        bool10 = ta_Arr_used >= 10**(-0.25)
        ta_Arr_used[ bool10 ] = 10**(-0.2500001)

    if Inside_Bicone_Arr is None : Inside_Bicone_Arr = np.ones( len(V_Arr) )

    Inside_Bicone_Arr = np.atleast_1d( Inside_Bicone_Arr )

    In_Bool_used = Inside_Bicone_Arr[ mask_good ]

    return V_Arr_used , logNH_Arr_used , ta_Arr_used , In_Bool_used , mask_good
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def  RT_f_esc( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr=None , MODE='Parametrization' , Algorithm='Intrepolation' , Machine_Learning_Algorithm='Tree' ):


    '''
        Return the Lyman alpha escape fraction for a given outflow properties.

        Parameters
        ----------
        Geometry : string
                   The outflow geometry to use: Options: 'Thins_Shell',
                   'Galactic_Wind' , 'Bicone_X_Slab'.


        wavelength_Arr : 1-D sequence of floats
                         Array with the wavelength vales where the line
                         profile is computed. The units are meters, i.e.,
                         amstrongs * 1.e-10.


        V_Arr : 1-D sequence of float
                Array with the expansion velocity of the outflow. The unit
                are km/s. 

        logNH_Arr : 1-D sequence of float
                    Array with the logarithim of the outflow neutral hydrogen
                    column density. The units of the colum density are in c.g.s,
                    i.e, cm**-2.

        ta_Arr : 1-D sequence of float
                 Array with the dust optic depth of the outflow. 

        Inside_Bicone_Arr : optional 1-D sequence of bool
                            An Array with booleans, indicating if the bicone is face-on
                            or edge-on. If True then the bicone is face-on. If false the
                            bicone is edge-on. The probability of being face on is
                            np.cos( np.pi/4 ).


        MODE : optional string
               Set the mode in which the escape fraction is computed. It can be:
                    Analytic        : it uses an analytic equation fitted to the output of the RT MC code.
                    Parametrization : it computes the escape fraction using a function that depends on the 
                                      dust optical depts as in Neufeld et al. 1990.
                    Raw             : it uses directly the output of the RT MC code.

                Default = 'Parametrization'


        Algorithm : optional string
                Set how the escape fraction is computed. If MODE='Analytic' then this varialbe is useless.
                    Intrepolation    : Direct lineal interpolation.
                    Machine_Learning : uses machine learning algorithms
        
                Default = 'Intrepolation'


        Machine_Learning_Algorithm : optial string
                Set the machine learning algorith used. Available:
                    Tree   : decision tree
                    Forest : random forest
                    KN     : KN

                Default = 'Tree'


            .. versionadded:: 0.0.3

        Returns
        -------
        lines_Arr : 1-D sequence of float
                    The Lyman alpha escape fraction for V_Arr[i] ,
                    logNH_Arr[i] , ta_Arr[i] , Inside_Bicone_Arr[i].
    '''

    assert MODE in [ 'Parametrization' , 'Raw' , 'Analytic'] , 'The requested mode ' + MODE + ' is not available. The modes supported are : Parametrization , Raw , Analytic' 

    assert Algorithm in [ 'Intrepolation' , 'Machine_Learning' ] , 'The requested algorithm ' + Algorithm + ' is not available. The algorithms supported are : Intrepolation , Machine_Learning' 

    assert Geometry in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab' ] , 'The requested geoemtry ' + Geometry + ' is not available. The geometries supported are : Thin_Shell , Galactic_Wind , Bicone_X_Slab'
    
    V_Arr_used , logNH_Arr_used , ta_Arr_used , In_Bool_used , mask_good = pre_treatment_f_esc( V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr , MODE ) 

    f_esc_Arr = np.zeros( len( mask_good ) ) * np.nan

    if MODE == 'Parametrization'  :

        if Algorithm == 'Intrepolation' :
            funtion_to_use = RT_f_esc_Interpolation_Parameters 

        if Algorithm == 'Machine_Learning':
            funtion_to_use = RT_f_esc_Machine_Parameter


    if MODE == 'Raw'  :

        if Algorithm == 'Intrepolation' :
            funtion_to_use = RT_f_esc_Interpolation_Values

        if Algorithm == 'Machine_Learning':
            funtion_to_use = RT_f_esc_Machine_Values


    if MODE == 'Analytic' :

        funtion_to_use = RT_f_esc_Analytic        


    f_esc_Arr[ mask_good ] = funtion_to_use( Geometry , V_Arr_used , logNH_Arr_used , ta_Arr_used , Inside_Bicone_Arr=In_Bool_used , Machine_Learning_Algorithm=Machine_Learning_Algorithm )

    return f_esc_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def define_RT_parameters():
     T4 = 1. # = 10000. / 1e4
     nu0 = 2.46777 * 1.e15 #3. * 10.**8 / (1215.67 * (10**(-10)))
     Vth = 12.85 * np.sqrt(T4) # lo he comentado porque sqrt(1) = 1
     Dv = Vth * nu0 *1. / ( 3 * (10**5))
     return nu0 , Dv
#==============================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def convert_x_into_lamda( x ):
     nu0 , Dv = define_RT_parameters() 
     return( 3. * 1.e8 / ( x * Dv + nu0)  )

def convert_lamda_into_x( lamda ):
     nu0 , Dv = define_RT_parameters() 
     return( (( 3. * 1.e8 / lamda) -nu0 ) / Dv     )
#==============================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def load_Grid_Line( Geometry , INSIDE_BICONE=None ):

    '''
        Return the dictionary with all the properties of the grid where the lines were run.

        Parameters
        ----------
        Geometry : string
                   The outflow geometry to use: Options: 'Thins_Shell',
                   'Galactic_Wind' , 'Bicone_X_Slab'.

        INSIDE_BICONE : optional boolean
                        This is useless if the geometry is not Bicone_X_Slab. 
                        If True then the bicone is face-on. If false the
                        bicone is edge-on. The probability of being face 
                        on is np.cos( np.pi/4 ).
                

        Returns
        -------
        loaded_model : Dictionary
                       This dictonary have all the information of the grid.
                       Entries:
                            'V_Arr'     : Array of velocity expansions used.[km/s]
                            'logNH_Arr' : Array of logarithm of the column density. [c.g.s.]
                            'logta_Arr' : Array of logarithm of the dust optical depth.
                            'x_Arr'     : Array of frequency in Doppler  units.
                            'Grid'      : Array with the output of the RT MC code LyaRT:
                                         
                                loaded_model['Grid'][i,j,k,:] has the line profile evaluated in loaded_model['x_Arr']
                                with outflow velocity loaded_model['V_Arr'][i] , logarithm of the neutral hydrogen 
                                column density loaded_model['logNH_Arr'][j] and logarithm of dust optical depth 
                                loaded_model['logta_Arr'][k]  
    '''

    assert Geometry in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab' ] , 'The requested geoemtry ' + Geometry + '  is not available. The geometries supported are : Thin_Shell , Galactic_Wind , Bicone_X_Slab'

    Geometry_Set = [ 'Thin_Shell'  , 'Galactic_Wind'  , 'Bicone_X_Slab' ]

    dirs_Set     = [ 'Thin_Shell/' , 'Galactic_Wind/' , 'Bicone_X_Slab/' ]

    geo_code     = [ 'Thin_Shell'  , 'Wind'           , 'Bicone_X_Slab' ]

    index = np.where( Geometry == np.array(Geometry_Set) )[0][0]

    filename_root = 'DATA/Dictonary_'+ geo_code[index] +'_Grid_Lines'

    if Geometry == 'Bicone_X_Slab':
        filename_root += '_In_Bicone_' + str(INSIDE_BICONE)

    filename = filename_root + '.npy'


    this_dir, this_filename = os.path.split(__file__)

    filename = os.path.join(this_dir, filename)

    loaded_model = np.load( filename ).item()

    return loaded_model
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_Lines_Arrays_3D_grid( V_Arr , logNH_Arr , logta_Arr , x_Arr , Grid_Dictionary ):

    Grid_Line = Grid_Dictionary['Grid']

    V_Arr_Grid = Grid_Dictionary['V_Arr']
    x_Arr_Grid = Grid_Dictionary['x_Arr']

    logNH_Arr_Grid = Grid_Dictionary['logNH_Arr']
    logta_Arr_Grid = Grid_Dictionary['logta_Arr']

    lines_Arr = np.zeros( len(V_Arr) * len( x_Arr ) ).reshape( len(V_Arr) , len( x_Arr ) )

    for i in range( 0 , len( V_Arr ) ):

        aux_line = Linear_3D_interpolator( V_Arr[i] , logNH_Arr[i] , logta_Arr[i] , V_Arr_Grid , logNH_Arr_Grid , logta_Arr_Grid , Grid_Line )

        axu_line_1 = np.interp( x_Arr , x_Arr_Grid , aux_line , left=0.0 , right=0.0 )

        Integral = np.trapz( axu_line_1 , x_Arr )

        lines_Arr[i] = np.absolute( axu_line_1 * 1. / Integral )

    return lines_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Interpolate_Lines_Arrays_3D_grid_MCMC( V_Value , logNH_Value , logta_Value , x_Arr , Grid_Dictionary ):

    Grid_Line = Grid_Dictionary['Grid']

    V_Arr_Grid = Grid_Dictionary['V_Arr']
    x_Arr_Grid = Grid_Dictionary['x_Arr']

    logNH_Arr_Grid = Grid_Dictionary['logNH_Arr']
    logta_Arr_Grid = Grid_Dictionary['logta_Arr']

    aux_line = Linear_3D_interpolator( V_Value , logNH_Value , logta_Value , V_Arr_Grid , logNH_Arr_Grid , logta_Arr_Grid , Grid_Line )

    axu_line_1 = np.interp( x_Arr , x_Arr_Grid , aux_line , left=0.0 , right=0.0 )

    Integral = np.trapz( axu_line_1 , x_Arr )

    axu_line_1 = np.absolute( axu_line_1 * 1. / Integral )

    return axu_line_1
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def pre_treatment_Line_profile_MCMC( Geometry , V_Value , logNH_Value , ta_Value ):

    bool1 = np.isfinite(     V_Value )
    bool2 = np.isfinite( logNH_Value )
    bool3 = np.isfinite(    ta_Value )

    Bool_good = bool1 * bool2 * bool3

    if Geometry in [ 'Thin_Shell' , 'Bicone_X_Slab' ]:
        if V_Value <= 100.0 and logNH_Value >= 20.5 :

            #aux_V = logNH_Value*-40 + 920.
            aux_V = logNH_Value * ( -100/1.5 ) + ( 100 - ( -100/1.5 * 20.5 ) )

            bool_aux = V_Value > aux_V

            if     bool_aux : V_Value = 100.0001
            if not bool_aux : logNH_Value = 20.4999999


    if V_Value <=   10.0 : V_Value =  10.000001
    if V_Value >= 1000.0 : V_Value = 999.999999

    if logNH_Value <    17.0 : logNH_Value =  17.000001
    if logNH_Value >=   22.0 : logNH_Value =  21.999999

    if ta_Value <   10**(-3.75 ) : ta_Value =  10**(-3.749999999)
    if ta_Value >=  10**(-0.125) : ta_Value =  10**(-0.125000001)

    return V_Value , logNH_Value , ta_Value , Bool_good
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def RT_Line_Profile_MCMC( Geometry , wavelength_Arr , V_Value , logNH_Value , ta_Value , DATA_LyaRT ):

    '''
        Return one and only one Lyman alpha line profile for a given outflow properties.
        This function is especial to run MCMCs or PSO.

        Parameters
        ----------
        Geometry : string
                   The outflow geometry to use: Options: 'Thins_Shell',
                   'Galactic_Wind' , 'Bicone_X_Slab'.

        wavelength_Arr : 1-D sequence of floats
                         Array with the wavelength vales where the line
                         profile is computed. The units are meters, i.e.,
                         amstrongs * 1.e-10.

        V_Value : float
                  Value of the expansion velocity of the outflow. The unit
                  are km/s. 

        logNH_Value : float
                      Value of the logarithim of the outflow neutral hydrogen
                      column density. The units of the colum density are in c.g.s,
                      i.e, cm**-2. 

        ta_Value : float
                 Value of the dust optic depth of the outflow. 

        DATA_LyaRT : Dictionay
                     This dictonary have all the information of the grid.
                     This dictionary can be loaded with the function : 
                     load_Grid_Line, for example:

                     DATA_LyaRT = load_Grid_Line( 'Thin_Shell' ) 

        Returns
        -------
        lines_Arr : 1-D sequence of float
                    The Lyman alpha line profile. 
    '''

    V_Value , logNH_Value , ta_Value , Bool_good = pre_treatment_Line_profile_MCMC( Geometry , V_Value , logNH_Value , ta_Value )

    if Bool_good :
        logta_Value = np.log10( ta_Value )

        x_Arr = convert_lamda_into_x( wavelength_Arr )

        line_Arr = Interpolate_Lines_Arrays_3D_grid_MCMC( V_Value , logNH_Value , logta_Value , x_Arr , DATA_LyaRT )

    if not Bool_good :

        line_Arr = np.ones( len(x_Arr) ) * np.nan

    return line_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def pre_treatment_Line_profile( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr ):

    V_Arr     = np.atleast_1d(     V_Arr )
    logNH_Arr = np.atleast_1d( logNH_Arr )
    ta_Arr    = np.atleast_1d(    ta_Arr )

    V_Arr     =     V_Arr.astype(float)
    logNH_Arr = logNH_Arr.astype(float)
    ta_Arr    =    ta_Arr.astype(float)

    bool1 = np.isfinite( V_Arr     )
    bool2 = np.isfinite( logNH_Arr )
    bool3 = np.isfinite( ta_Arr    )

    mask_good = bool1 * bool2 * bool3

    assert sum( mask_good ) != 0 , 'All the V-logNH-ta combinations are np.nan, -np.inf or np.inf'

    V_Arr_used     =     V_Arr[ mask_good ]
    logNH_Arr_used = logNH_Arr[ mask_good ]
    ta_Arr_used    =    ta_Arr[ mask_good ]

    #============================================#
    if Geometry in ['Thin_Shell' , 'Bicone_X_Slab']:
        bool1 = V_Arr_used < 100.0
        bool2 = logNH_Arr_used >= 20.5

        aux_V_arr = logNH_Arr_used * ( -100/1.5 ) + ( 100 - ( -100/1.5 * 20.5 ) )

        bool_aux = V_Arr_used > aux_V_arr

        V_Arr_used[     bool1 * bool2 *  bool_aux ] = 100.000001
        logNH_Arr_used[ bool1 * bool2 *~ bool_aux ] = 20.499999
    #============================================#


    bool5 = V_Arr_used <= 10
    V_Arr_used[ bool5 ] = 10.000001

    bool6 = V_Arr_used >= 1000
    V_Arr_used[ bool6 ] = 999.9999

    bool7 = logNH_Arr_used <= 17.0
    logNH_Arr_used[ bool7 ] = 17.0000001

    bool8 = logNH_Arr_used >= 22.0
    logNH_Arr_used[ bool8 ] = 21.9999

    bool9 = ta_Arr_used <= 10**(-3.75)
    ta_Arr_used[ bool9 ] = 10**(-3.74999999)

    bool10 = ta_Arr_used >= 10**(-0.125)
    ta_Arr_used[ bool10 ] = 10**(-0.125000000001)


    if Inside_Bicone_Arr is None : Inside_Bicone_Arr = np.ones( len(V_Arr) )
    else: Inside_Bicone_Arr = np.atleast_1d( Inside_Bicone_Arr )


    In_Bool_used = Inside_Bicone_Arr[ mask_good ]

    return V_Arr_used , logNH_Arr_used , ta_Arr_used , In_Bool_used , mask_good
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def RT_Line_Profile( Geometry , wavelength_Arr , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr=None ):

    '''
        Return the Lyman alpha line profile for a given outflow properties.
        
        Parameters
        ----------
        Geometry : string
                   The outflow geometry to use: Options: 'Thins_Shell',
                   'Galactic_Wind' , 'Bicone_X_Slab'.
        
        wavelength_Arr : 1-D sequence of floats
                         Array with the wavelength vales where the line 
                         profile is computed. The units are meters, i.e.,
                         amstrongs * 1.e-10.
        
        V_Arr : 1-D sequence of float 
                Array with the expansion velocity of the outflow. The unit
                are km/s. 
        
        logNH_Arr : 1-D sequence of float
                    Array with the logarithim of the outflow neutral hydrogen 
                    column density. The units of the colum density are in c.g.s,
                    i.e, cm**-2. 
        
        ta_Arr : 1-D sequence of float
                 Array with the dust optic depth of the outflow. 

        Inside_Bicone_Arr : optional 1-D sequence of bool
                            This is useless if the geometry is not Bicone_X_Slab.
                            An Array with booleans, indicating if the bicone is face-on 
                            or edge-on. If True then the bicone is face-on. If false the
                            bicone is edge-on. The probability of being face on is 
                            np.cos( np.pi/4 ).
        
            .. versionadded:: 0.0.3
        
        Returns
        -------
        lines_Arr : 2-D sequence of float
                    The Lyman alpha line profiles. lines_Arr[i] is the line profile 
                    computed at the wavelengths wavelength_Arr for wich V_Arr[i] , 
                    logNH_Arr[i] , ta_Arr[i] , Inside_Bicone_Arr[i].
    '''

    assert Geometry in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab' ] , 'The requested geoemtry ' + Geometry + ' is not available. The geometries supported are : Thin_Shell , Galactic_Wind , Bicone_X_Slab'

    V_Arr             = np.atleast_1d(     V_Arr )
    logNH_Arr         = np.atleast_1d( logNH_Arr )
    ta_Arr            = np.atleast_1d(    ta_Arr )

    x_Arr = convert_lamda_into_x( wavelength_Arr )

    lines_Arr = np.zeros( len(V_Arr) * len( x_Arr ) ).reshape( len(V_Arr) , len( x_Arr ) ) * np.nan
       
    V_Arr_used , logNH_Arr_used , ta_Arr_used , In_Bool_used , mask_good = pre_treatment_Line_profile( Geometry , V_Arr , logNH_Arr , ta_Arr , Inside_Bicone_Arr )


    logta_Arr_used = np.log10( ta_Arr_used )

    ##############################

    if Geometry in [ 'Thin_Shell'  , 'Galactic_Wind'  ] :

        DATA_LyaRT = load_Grid_Line( Geometry )

        tmp_lines_Arr = Interpolate_Lines_Arrays_3D_grid( V_Arr_used , logNH_Arr_used , logta_Arr_used , x_Arr , DATA_LyaRT )

    ##############################

    if Geometry in [ 'Bicone_X_Slab' ] :

        assert not Inside_Bicone_Arr is None , 'Error. Human is dead. Mismatch. \nIf the goemetry is Bicone_X_Slab then it is compulsory to define Inside_Bicone_Arr when colling this function.' 

        Inside_Bicone_Arr =  np.atleast_1d( Inside_Bicone_Arr )

        tmp_lines_Arr = np.zeros( len( V_Arr_used ) * len( x_Arr ) ).reshape( len( V_Arr_used ) , len( x_Arr ) ) 

        DATA_LyaRT_in  = load_Grid_Line( Geometry , INSIDE_BICONE=True  )
        DATA_LyaRT_out = load_Grid_Line( Geometry , INSIDE_BICONE=False )

        lines_Arr_in  = Interpolate_Lines_Arrays_3D_grid( V_Arr_used[ In_Bool_used] , logNH_Arr_used[ In_Bool_used] , logta_Arr_used[ In_Bool_used] , x_Arr , DATA_LyaRT_in  ) 
        lines_Arr_out = Interpolate_Lines_Arrays_3D_grid( V_Arr_used[~In_Bool_used] , logNH_Arr_used[~In_Bool_used] , logta_Arr_used[~In_Bool_used] , x_Arr , DATA_LyaRT_out ) 

        tmp_lines_Arr[ In_Bool_used] = lines_Arr_in
        tmp_lines_Arr[~In_Bool_used] = lines_Arr_out

    lines_Arr[ mask_good ] = tmp_lines_Arr

    return lines_Arr
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Print_the_grid_edges():

    print ''
    print '    Hi,'
    print ''
    print '    The expanssion velocity V_exp and neutral hydrogen column density logNH are the same in the escape fraction and line profile grids. However, the optical depth of dust tau_a is different.'
    print ''
    print '    V_exp [ km/s ] = [ 0 , 10 , ... , 90 , 100 , 150 , 200 , ... , 950 , 1000 ]'
    print ''
    print '    For V_exp <  100 km/s the logNH [ cm**-2 ] = [ 17.0 , 17.25 , ... , 20.25 , 20.5 ]'
    print '    '
    print '    For V_exp >= 100 km/s the logNH [ cm**-2 ] = [ 17.0 , 17.25 , ... , 21.75 , 22.0 ]'
    print '    '
    print '    For the escape fraction : tau_a = [ -3. , -2. , -1.5 , -1.0 , -0.75 , -0.5 , -0.25 , -0.0 ]'
    print '    '
    print '    For the line profile    : tau_a = [  -0.125 , -0.25 , -0.375 , -0.5 , -0.625 , -0.75 , -0.875 , -1.0 , -1.125 , -1.25 , -1.375 , -1.5 , -1.75 , -2.0 , -2.25 , -2.5 , -2.75 , -3.0 , -3.25 , -3.5 , -3.75 ]'
    print ''
    print '    Have a nice day!'
    print '    El. PSY. CONGROO.'
    print ''

    return
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Test_1( ):
    print '\nChecking if all the files are found...',
    
    bool_files = Check_if_DATA_files_are_found()
    
    print 'Done!'
    
    if bool_files :
        print '    Every file was found. that is great!'
    if not bool_files :
        print '    Missing files.... Let us download them... ;)'
    
        Download_data()
   
    print '\n Now that we are sure that the data is downloaded in your machine...'

    print '\n Let us check every different configuration for computing the escape fraction and the line profiles.'
 
    Geometry_set = [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab' ]
    
    ML_codes_set = [ 'Tree' , 'Forest' , 'KN' ]
    
    MODE_set = [ 'Parametrization' , 'Raw' , 'Analytic' ]
    
    Algorithm_set = [ 'Intrepolation' , 'Machine_Learning' ]
    
    # Primero vamos a checkear que funciona las fracciones de escape
    
    N_points = int( 1e4 )
    
    V_Arr     = np.random.rand( N_points ) * 1000   + 0.0
    logNH_Arr = np.random.rand( N_points ) *   5   + 17.0
    logta_Arr = np.random.rand( N_points ) *   4.5 -  4.0
    
    In_Arr = np.random.rand( N_points ) > 0.5
    
    print '\nComputing', N_points , 'random configurations of escape fraction with each algorithms...\n'
    
    for Geo in Geometry_set:
    
        for Mod in MODE_set :
    
            if not Mod in [ 'Analytic' ]:
    
                for Algo in Algorithm_set:
    
                    if Algo in [ 'Intrepolation' , 'Machine_Learning' ]:
    
                        if Algo == 'Machine_Learning' :
    
                            for machine in ML_codes_set :
    
    
                                try:
                                    print '      Running : ' , Geo , Mod , Algo , machine ,
                                    fff = RT_f_esc( Geo , V_Arr , logNH_Arr , 10**logta_Arr , Inside_Bicone_Arr=In_Arr , MODE=Mod , Algorithm=Algo , Machine_Learning_Algorithm=machine)
                                    assert np.sum( np.isnan( fff ) ) == 0
                                    print '--> Success!!'
                                except:
                                    print '--> ERROR. HUMAN IS DEAD. MISMATCH!!'
    
                        if Algo != 'Machine_Learning' :
    
    
                                try:
                                    print '      Running : ' , Geo , Mod , Algo ,
                                    fff = RT_f_esc( Geo , V_Arr , logNH_Arr , 10**logta_Arr , Inside_Bicone_Arr=In_Arr , MODE=Mod , Algorithm=Algo )
                                    assert np.sum( np.isnan( fff ) ) == 0
                                    print '--> Success!!'
    
                                except:
                                    print '--> ERROR. HUMAN IS DEAD. MISMATCH!!'
    
            if Mod in [ 'Analytic' ]:
    
    
                                try:
                                    print '      Running : ' , Geo , Mod ,
                                    fff = RT_f_esc( Geo , V_Arr , logNH_Arr , 10**logta_Arr , MODE=Mod )
                                    assert np.sum( np.isnan( fff ) ) == 0
                                    print '--> Success!!'
    
                                except:
                                    print '--> ERROR. HUMAN IS DEAD. MISMATCH!!'
    
    
    
    N_points = int( 1e3 )
    
    print '\nComputing', N_points , 'random configurations of line profile  with each algorithms...\n'
    
    V_Arr     = np.random.rand( N_points ) * 1000   + 0
    logNH_Arr = np.random.rand( N_points ) *   5   + 17.0
    logta_Arr = np.random.rand( N_points ) *   5.5 -  4.75
    
    In_Arr = np.random.rand( N_points ) > 0.5
    
    wavelength_Arr = np.linspace( 1215.68 - 20 , 1215.68 + 20 , 1000 ) * 1e-10
    
    RUN_TEST_Lines = True
    if RUN_TEST_Lines :
        for Geo in Geometry_set:
    
            print '      Running : ' , Geo ,
    
            try:
                qq = RT_Line_Profile( Geo , wavelength_Arr , V_Arr , logNH_Arr , 10**logta_Arr , Inside_Bicone_Arr=In_Arr )
                assert np.sum( np.isnan( qq ) ) == 0
                print '--> Success!!'
    
            except:
                print '--> ERROR. HUMAN IS DEAD. MISMATCH!!'

    return
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Test_2( ):

    from pylab import *

    print '\n Let us make some plots. This will show you just a glimpse of what LyaRT;Grid can do. Just wait for it...'

    # Plot some nice line profiles

    print '\n    Plotting some line profiles...'

    wavelength_Arr = np.linspace( 1215.68 - 20 , 1215.68 + 20 , 1000 ) * 1e-10

    V_Arr = np.array( [ 10 , 50 , 100 , 200 , 300 ] )

    logNH_Arr = np.array( [ 20.0 ] * len( V_Arr ) )

    logta_Arr = np.array( [ -1. ] * len( V_Arr ) )

    Inside_Bicone_Arr = np.zeros( len(V_Arr) ) == 0

    cm = get_cmap( 'rainbow' )
    
    for geo in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab' ]:

        qq = RT_Line_Profile( geo , wavelength_Arr , V_Arr , logNH_Arr , 10.**logta_Arr , Inside_Bicone_Arr=Inside_Bicone_Arr ) 

        figure()

        ax_ax = subplot(111)

        for i in range( 0 ,len( V_Arr ) ):

            ax_ax.plot( wavelength_Arr*1e10 , qq[i] , color=cm( i*1./( len(V_Arr) -1 ) ) , label=r'$\rm V_{exp} = '+ str(V_Arr[i]) +'km/s$ ' , lw=2 )

        texto = r'$\rm N_{H} = 10^{20} cm^{-2}$' + '\n' + r'$\rm \tau_{a} = 0.1$'

        ax_ax.text( .95 , 0.45 , texto , verticalalignment='top', horizontalalignment='right', transform=ax_ax.transAxes, fontsize=20 )
        
        ax_ax.set_title( r'$\rm Geometry = $' + geo , size=20 )

        ax_ax.set_ylabel( r'$\rm Flux [a.u.]$' , size=20 )
        ax_ax.set_xlabel( r'$\rm Wavelength [\AA]$' , size=20 )

        ax_ax.set_xlim( 1212.5 , 1222.5 )

        ax_ax.legend(loc=0)

    print '\n    Plotting some escape fractions...'

    logta_Arr = np.linspace( -2 , 0.5 , 20 )

    logNH_Arr = [20.0] * len( logta_Arr )

    for geo in [ 'Thin_Shell' , 'Galactic_Wind' , 'Bicone_X_Slab' ]  :

        figure()

        ax_ax = subplot(111)

        for i in range( 0 , len(V_Arr) ):

            V_Arr_tmp = [ V_Arr[i] ] * len( logta_Arr )

            Inside_Bicone_Arr = np.zeros( len( logta_Arr ) ) == 0

            f_esc = RT_f_esc( geo , V_Arr_tmp , logNH_Arr , 10**logta_Arr , Inside_Bicone_Arr=Inside_Bicone_Arr)
        
            ax_ax.plot( logta_Arr , f_esc , color=cm( i*1./( len(V_Arr) -1 ) ) , label=r'$\rm V_{exp} = '+ str(V_Arr[i]) +'km/s$ ' , lw=2 )

            Inside_Bicone_Arr = np.zeros( len( logta_Arr ) ) == 1

            f_esc = RT_f_esc( geo , V_Arr_tmp , logNH_Arr , 10**logta_Arr , Inside_Bicone_Arr=Inside_Bicone_Arr)

            ax_ax.semilogy( logta_Arr , f_esc , '--' , color=cm( i*1./( len(V_Arr) -1 ) ) , lw=2 )

        ax_ax.set_xlabel( r'$\rm \log \tau_a$' , size=20 )
        ax_ax.set_ylabel( r'$f_{\rm esc} ^{\rm Ly \alpha} $' , size=20 )

        texto = r'$\rm N_{H} = 10^{20} cm^{-2}$'

        ax_ax.text( .5 , 0.05 , texto , verticalalignment='bottom', horizontalalignment='left', transform=ax_ax.transAxes, fontsize=20 )

        ax_ax.set_title( r'$\rm Geometry = $' + geo , size=20 )

        legend( loc=0 )

    show()

    return
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
def Test_Installation( Make_Plots=True ):

    import warnings
    warnings.filterwarnings("ignore")

    Test_1()

    if Make_Plots : Test_2()

    return
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#
#====================================================================#

if __name__ == '__main__':
    pass


