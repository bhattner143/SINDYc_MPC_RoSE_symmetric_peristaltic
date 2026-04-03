#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 19:37:54 2020

@author: dipankarbhattacharya
"""
from QSR2Layer_ADC_SINDYc_discrete_using_PySINDY_MPC_QSR_2Layer import*
from drawnow import drawnow

# =============================================================================
#Main function: MPC Loop
# =============================================================================

def main(QSR_DoubleLayer_Obj_SINDYc_SI): 
    try:           
        # =============================================================================
        # various prediction horizon loop
        # =============================================================================
        # Choose prediction horizon over which the optimization is performed
        x0=np.array([70])#x0_robot_adc_valid
        Nvec=np.array([4])
        
        for i in range(Nvec.shape[0]):
            
            Ts          = .1              # Sampling time
            N           = Nvec[i]          # Control / prediction horizon (number of iterations)
            Duration    = 100              # Run control for 100 time units
            Nvar        = 1
            Q           = np.array([1])            # State weights
            #Q=Q[np.newaxis,:]
            R           = 0.5 #0.5;         # Control variation du weights
            Ru = 0.5#0.01                 # Control weights
        #    B = np.array([0,1])                    # Control vector (which state is controlled)
        #    B=B[:,np.newaxis]
        #    #C = eye(Nvar)                  # Measurement matrix
        #    D = 0                          # Feedforward (none)
            x0n=x0.T                       # Initial condition
            uopt0 = 30                     # Set initial control input to thirty
            Ton      = 30                  # Time when control starts
            
            # Reference trajectory, which shall be achieved
            xref0 = 0*np.ones((1,int(Ton/Ts))).T #Initial part of the reference where the control is off
            xref1 = 60*np.ones((1,100)).T
            xref2 = 70*np.ones((1,250)).T
            xref4 = 75*np.ones((1,200)).T
            xref3 = 65*np.ones((1,151)).T
            
            args=(xref0,xref1,xref2,xref3,xref4)
            xref_all=np.concatenate(args)
            assert xref_all.shape[0]-1 == int(Duration/Ts), 'size of xref_all must be equal to int(Duration/Ts)'
            
            x        = x0n
            x=np.asarray(x)
            
            uopt     = uopt0*np.ones(N)
            uopt=np.asarray(uopt)
            
            xHistory = x       # Stores state history
            uHistory = uopt[0] # Stores control history
            tHistory = np.zeros((1))       # Stores time history
            rHistory = xref1   # Stores reference (could be trajectory and vary with time)
            x_Tof_History=np.zeros((3))
            

            
            funval=np.zeros((int(Duration/Ts)))
            
            start_time = time.time()
        
            # RoSE Actuation parameters
            BaseLinePress=np.zeros((1,12))
            ScalingFact=1.0
            num_rose_layers=12
            
            #Live CSV file initialization
            fieldnames=['xHistory','x_Tof_History_1','x_Tof_History_2','uHistory','tHistory','rHistory','JHistory']
            
            with open('DataFiles/data.csv', 'w') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()
                
            #Initialize TOF 
            t1 = timedelta(minutes = 0, seconds = 0, microseconds=0)
            
            if QSR_DoubleLayer_Obj_SINDYc_SI.Flag_UseADC is True and QSR_DoubleLayer_Obj_SINDYc_SI.Flag_UseIOExpander is True:
                
                TOFADCPer2dArray=np.array([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]).T
                range_mm_array=QSR_DoubleLayer_Obj_SINDYc_SI.GenerateDisplacementDataFromTOF(t1)
    
                #Design real-time filter
                b,z=RoSE_actuation_protocol.Filter_RealTime_design(numtaps=50 )
                z=(z,z)
                TOF2dArray_mean_filtered_stacked=np.array([[],[],[]]).T
            
            starttime=time.time()
            # =============================================================================
            #Function for live plotting
            # =============================================================================
            
            plt.ion()  # enable interactivity
            f7 = plt.figure()  # make a figure
            
            def make_fig():
                
                # plt.plot(x_vals, y_vals)  # I think you meant this
                # plt.plot(x_vals,u_vals)
                ax1 = f7.add_subplot(211)
                
                ax1.plot(xHistory[:,0],
                    color='blue',
                    linewidth=1.5,
                    linestyle='-')
                
                ax1.plot(rHistory[:,0],
                    color='k',
                    linewidth=1.5,
                    linestyle='--')
                ax1.set_ylim([0,150])
                ax1.set_xlim([0,1000])
                
                ax2 = f7.add_subplot(212)
                ax2.plot(uHistory[:,0],
                    color='red',
                    linewidth=1.5,
                    linestyle='-')
                ax2.set_ylim([0,150])
                ax2.set_xlim([0,1000])
      
            # =============================================================================
            # MPC loop
            # =============================================================================
            for ct in range(int(Duration/Ts)):
                
                if ct*Ts>Ton:   # Turn control on
                    
                    if ct*Ts==Ton+Ts:
                        print('Start Control.')
                        
                
                    # Set references
                    #xref = np.asarray(xref1)
                    xref = xref_all[ct]
                    
                    # Constraints on control optimization
                    if xref==60:
                        LB=22.5 #Lower bound
                        UB=120  #Upper bound
                    elif xref==70:
                        LB=57.5
                        UB=120#77.5
                    elif xref==75:
                        LB=82.5
                        UB=120#92.5
                    elif xref==65:
                        LB=37.5
                        UB=120
                        
                    
                    obj=QSR_DoubleLayer_Obj_SINDYc_SI
                                   
                    uopt=minimize(obj.RobotObjectiveFCN,
                                  uopt,
                                  method='SLSQP',
                                  args=(x[0],Ts,N,xref,uopt[0],np.diag(Q),R,Ru),
                                  tol=0.1,
                                  options={'ftol': 0.1, 'disp': False},
                                  # constraints=cons,
                                  bounds=[(LB,UB),(LB,UB),(LB,UB),(LB,UB)]
                                  )
                    # pdb.set_trace()
                    
                    funval[ct]=uopt.fun
                    
                    if np.absolute(x[0])>700:
                        break
                    
                    uopt=uopt.x
                    
                else:    #% If control is off
                    
                    if ct*Ts==0:
                        print('Control is off')
                        
                    uopt=uopt0*np.ones((N))
                    xref=-1000*np.ones((1))
#                    pdb.set_trace()
                    
                # =============================================================================
                #Integrate system: Apply control & Step one timestep forward
                # =============================================================================
                if x.ndim==0:
                        x = x[np.newaxis]   
                        
                #Apply input to RoSE
                uopt0_all_layers=uopt[0]*np.ones((1,num_rose_layers))
                QSR_DoubleLayer_Obj_SINDYc_SI.mergeDACadd2DataAndSend(uopt0_all_layers,
                                              0,
                                              BaseLinePress,
                                              ScalingFact,
                                              num_rose_layers,0)
                
                #Generate ADC output from RoSE
                if QSR_DoubleLayer_Obj_SINDYc_SI.Flag_UseIOExpander is True \
                and QSR_DoubleLayer_Obj_SINDYc_SI.Flag_UseADC is True:
                    
                    
                    pressure_kpa_array=QSR_DoubleLayer_Obj_SINDYc_SI.GeneratePressureDataFromADC(NoOfADC=12)
                    
                    range_mm_array=QSR_DoubleLayer_Obj_SINDYc_SI.GenerateDisplacementDataFromTOF(t1)
                    
                    range_mm_array_mean=np.array([range_mm_array[0],
                                              range_mm_array[1:6].mean(axis=0),
                                              range_mm_array[7:].mean(axis=0)])
                    range_mm_array_filtered, z= RoSE_actuation_protocol.Filter_RealTime_apply(range_mm_array_mean[1:],
                                                                                          b,z)
                    range_mm_array_filtered=range_mm_array_filtered[...,np.newaxis].T
                    range_mm_array_filtered=np.concatenate((np.array([range_mm_array[0]]),
                                              range_mm_array_filtered[0]))
                    
#                    TOFADCPer2dArray=np.concatenate((TOFADCPer2dArray,range_pressure_peristalsis_array[np.newaxis,:]))
                                        
                    
                x = np.array([[pressure_kpa_array[3]]])
                
                t_History=tHistory[-1]+Ts/1
                xHistory=np.vstack((xHistory,x))
                x_Tof_History=np.vstack((x_Tof_History,range_mm_array_filtered))
                uHistory=np.vstack((uHistory,uopt[0]))
                tHistory = np.vstack((tHistory,t_History))
                rHistory = np.vstack((rHistory,xref))
            # =============================================================================
            # Live CSV data writing    
            # =============================================================================
                with open('DataFiles/data.csv', 'a') as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    
        
                    HistoryDict={'xHistory':x[0,0],
                                 'x_Tof_History_1':range_mm_array_filtered[1],
                                 'x_Tof_History_2':range_mm_array_filtered[2],
                                 'uHistory':uopt[0],
                                 'tHistory':tHistory[-1,0],
                                 'rHistory':xref[0],
                                 'JHistory':funval[ct]}
                    
                    csv_writer.writerow(HistoryDict)
                    
            # =============================================================================
            # Live-plotting            
            # =============================================================================
#                drawnow(make_fig)
                    
            print("--- %s seconds ---" % (time.time() - start_time))
                
            # =============================================================================
            # Create a Dicionary for storing the data history
            # =============================================================================
            HistoryDict={'xHistory':xHistory,
                         'x_Tof_History':x_Tof_History,
                         'uHistory':uHistory,
                         'tHistory':tHistory,
                         'rHistory':rHistory,
                         'JHistory':funval}
            
            # Clear input to the RoSE:Actuate it with 0 kPa        
            QSR_DoubleLayer_Obj_SINDYc_SI.RobotNoActuation() 
        
        # f1.savefig('Plots/MPC/MPC_N_'+str(i)+'.png', bbox_inches='tight',dpi=300)
                    
    
    except KeyboardInterrupt:
        
        # =============================================================================
        # Clear input to the RoSE:Actuate it with 0 kPa        
        # =============================================================================
        QSR_DoubleLayer_Obj_SINDYc_SI.RobotNoActuation()
        print('Ctrl C Pressed')
        
    except:
        
        # =============================================================================
        # Clear input to the RoSE:Actuate it with 0 kPa        
        # =============================================================================
        QSR_DoubleLayer_Obj_SINDYc_SI.RobotNoActuation() 
#        
if __name__=='__main__':
    
    # =============================================================================
    #Modeling and prediction 
    # =============================================================================
    try:
        QSR_DoubleLayer_Obj_SINDYc_SI
    except NameError:
        print('Model does not exist, so generating it\n')
        QSR_DoubleLayer_Obj_SINDYc_SI=sindyc_model()
        print('Modeling done\n')
    else:
        print('Using existing model\n')   
        main(QSR_DoubleLayer_Obj_SINDYc_SI)
else:
    print('Run from another module')        