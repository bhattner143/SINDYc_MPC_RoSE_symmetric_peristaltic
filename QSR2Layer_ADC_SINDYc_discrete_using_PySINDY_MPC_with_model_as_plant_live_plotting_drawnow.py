#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 19:37:54 2020

@author: dipankarbhattacharya
"""
from QSR2Layer_ADC_SINDYc_discrete_using_PySINDY_MPC_with_model_as_plant import*
from drawnow import drawnow


# =============================================================================
#Main function: MPC Loop
# =============================================================================
def main():
    
    try:
        
        # =============================================================================
        #Modeling and prediction 
        # =============================================================================
        QSR_DoubleLayer_Obj_SINDYc_SI=sindyc_model()
    
        # Choose prediction horizon over which the optimization is performed
        x0=np.array([100])#x0_robot_adc_valid
        Nvec=np.array([1])
        
        for i in range(Nvec.shape[0]):
            
            Ts          = 0.1              # Sampling time
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
            x0n=x0.T#[100; 50];             # Initial condition
            uopt0 = 70                     # Set initial control input to thirty
            
            # Constraints on control optimization
            LB = 100#[] #-100*ones(N,1);        # Lower bound of control input
            UB = 115           # Upper bound of control input
            
            # Reference state, which shall be achieved
            xref1 = np.array([75]).T
            
            x        = x0n
            x=np.asarray(x)
            
            Ton      = 30      # Time when control starts
            uopt     = uopt0*np.ones(N)
            uopt=np.asarray(uopt)
            
            xHistory = x       # Stores state history
            uHistory = uopt[0] # Stores control history
            tHistory = np.zeros((1))       # Stores time history
            rHistory = xref1   # Stores reference (could be trajectory and vary with time)
            
            #pdb.set_trace()
            
            bound_optVar=[(LB,UB)]
            
            funval=np.zeros((int(Duration/Ts)))
            
            start_time = time.time()
        
            fieldnames=['xHistory','uHistory','tHistory','rHistory','JHistory']
            
            with open('DataFiles/data.csv', 'w') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()
        
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
            
            #MPC loop
            for ct in range(int(Duration/Ts)):
                
                if ct*Ts>30:   # Turn control on
                    
                    if ct*Ts==Ton+Ts:
                        print('Start Control.')
                
                    # Set references
                    xref = np.asarray(xref1)
    #                pdb.set_trace()
                    
                    obj=QSR_DoubleLayer_Obj_SINDYc_SI
                    
    
                    
                    uopt=minimize(obj.RobotObjectiveFCN,
                                  uopt,
                                  method='SLSQP',
                                  args=(x[0],Ts,N,xref,uopt[0],np.diag(Q),R,Ru),
                                  tol=0.1,
                                  options={'ftol': 0.1, 'disp': False},
                                  # constraints=cons,
                                  bounds=[(LB,UB)]
                                  )
                    # pdb.set_trace()
                    
                    funval[ct]=uopt.fun
                    
                    if np.absolute(x[0])>700:
                        break
                    
                    uopt=uopt.x
                    
                else:    #% If control is off
                    
                    uopt=uopt0*np.ones((N))
                    xref=-1000*np.ones((1))
                    
                # Integrate system: Apply control & Step one timestep forward
                if x.ndim==0:
                        x = x[np.newaxis]    
                        
                x = QSR_DoubleLayer_Obj_SINDYc_SI.model.predict(x,uopt[0])
                
                t_History=tHistory[-1]+Ts/1
                xHistory=np.vstack((xHistory,x))
                uHistory=np.vstack((uHistory,uopt[0]))
                tHistory = np.vstack((tHistory,t_History))
                rHistory = np.vstack((rHistory,xref))
            # =============================================================================
            # Live CSV data writing    
            # =============================================================================
                with open('DataFiles/data.csv', 'a') as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    
        
                    HistoryDict={'xHistory':x[0,0],
                     'uHistory':uopt[0],
                     'tHistory':tHistory[-1,0],
                     'rHistory':xref[0],
                     'JHistory':funval[ct]}
                    
                    csv_writer.writerow(HistoryDict)
                    
                drawnow(make_fig)
                    
                    
                # data = pd.read_csv('DataFiles/data.csv')
                # x=data['xHistory']
            
                # plt.cla()
                # y_vals.append(x[0,0])
                # plt.plot(x[0,0],label='Channel 1')
            

                    
        
                      
            print("--- %s seconds ---" % (time.time() - start_time))
                
            # =============================================================================
            # Create a Dicionary for storing the data history
            # =============================================================================
            HistoryDict={'xHistory':xHistory,
                         'uHistory':uHistory,
                         'tHistory':tHistory,
                         'rHistory':rHistory,
                         'JHistory':funval}
            
            
            # # =============================================================================
            # #     Show results
            # # =============================================================================
            # tspan=QSR_DoubleLayer_Obj_SINDYc_SI.DataDictionary['tspan_valid']
            # f1 = plt.figure(num=7,figsize=(2.5, 1.5))
            # ax1 = f1.add_subplot(111)
             
            # ax1.plot(np.array([Ton+tspan[0],Ton+tspan[0]]),np.array([15,260]),
            #           color='limegreen',
            #             alpha=0.8,
            #             linestyle='--',
            #             linewidth=1.5)
            
            # ax1.plot(tHistory+tspan[0],np.zeros((tHistory.shape[0])),
            #         color='k',
            #         linewidth=1.5,
            #         linestyle='--')
            
            # ax1.plot(tHistory+tspan[0],xref1[0]*np.ones((tHistory.shape[0])),
            #         color='skyblue',
            #         linewidth=1.5,
            #         linestyle='--')
            
            # ph0 = ax1.plot(tHistory+tspan[0],xHistory[:,0],
            #             color='darkblue',
            #             linewidth=1.5,
            #             linestyle='-',
            #             label='Output')
            
            # ph2 = ax1.plot(tHistory+tspan[0],uHistory[:,0],
            #             color='brown',
            #             linewidth=1.5,
            #             linestyle='-',
            #             label='Control (u)')
            
            # ax1.legend(
            #                     ncol=1,
            #                     prop={'size': 12},
            # #                   mode="expand", 
            #                     borderaxespad=0,
            #                     handlelength=1,
            #                     labelspacing=0.1,
            #                     columnspacing=0.2,
            #                     loc=0,
            #                     frameon=False)
            
            # handles, labels = ax1.get_legend_handles_labels()
            
            # ax1.set_xlabel(r"Time", size=12)
            # ax1.set_ylabel(r"Population Size", size=12)
            # plt.show()
            
            # f1.savefig('Plots/MPC/MPC_N_'+str(i)+'.png', bbox_inches='tight',dpi=300)
            
            
            
    except KeyboardInterrupt:
        print('Ctrl C Pressed')
        
if __name__=='__main__':
    main()