#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:12:31 2019

@author: dipankarbhattacharya
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  4 12:57:15 2019

@author: dbha483
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:45:13 2018

@author: dipankarbhattacharya
"""
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.parser import parse
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

plt.close("all")


def cls():
    print("\n" * 50)


# clear Console
cls()


class stentMigration:
    def __init__(self, path, stentName, name):
        self.path = path
        self.stentName = stentName
        self.name = name
        self.currWorkDir = os.getcwd()
        os.chdir(
            self.path
        )  # whenever we will create an instance of the class, the path will be set for it
        self.DataFrameExtractedDict = {}
        self.ArrayDict = {}
        self.MeanMigrationInfoDict = {}
        self.GradMeanMigrationInfoDict = {}

    def find_all(self):
        #        pdb.set_trace()
        self.files = [
            files[index]
            for root, dirs, files in sorted(os.walk(self.path, topdown=True))
            for index in range(len(files))
            if self.name in files[index]
        ]

    # =============================================================================
    # Method: Read csv files and create a dictinonary of arrays
    # =============================================================================
    def FileReading(self, indexFileToRead):
        #        pdb.set_trace()
        for index in range(len(indexFileToRead)):

            if self.name == "TOF" or self.name == "QSR":
                filename1 = [
                    self.files[indexFileToRead[index]]
                ]  # ,self.files[indexFileToRead[1]],self.files[indexFileToRead[2]]]
                csv = pd.read_csv(filename1[0], sep=",", header=None)

                self.DataFrameExtractedDict[filename1[0].split(".")[0]] = csv

                TempArray = csv.to_numpy(dtype=np.dtype)
                self.ArrayDict[filename1[0].split(".")[0]] = np.array(
                    TempArray[2:, 1:], dtype="f"
                )
        # Returning the working directory to the  current working directory
        os.chdir(self.currWorkDir)

    # =============================================================================
    # Find mean and sd
    # =============================================================================

    def FindMeanSDAndRelDisp(self, *rangeDef):
        #       pdb.set_trace()
        #       MeanMigrationInfoArray=np.array()

        if not rangeDef:

            rangeDef = [
                self.ArrayDict[value].shape[0] for _, value in enumerate(self.ArrayDict)
            ]
            rangeDef = tuple([rangeDef])

        for c, value in enumerate(self.ArrayDict):

            TempArrayMean = np.mean(
                self.ArrayDict[value][0 : rangeDef[0][c], 2:], axis=1
            )

            MeanMigrationInfoArray = np.vstack(
                (
                    self.ArrayDict[value][0 : rangeDef[0][c], 0:2].T,
                    (TempArrayMean - TempArrayMean[0]),
                    np.std(self.ArrayDict[value][0 : rangeDef[0][c], 2:], axis=1),
                )
            ).T

            self.MeanMigrationInfoDict[value] = MeanMigrationInfoArray

    # =============================================================================
    # Compute a Gradient dictionasry of the MeanMigrationInfoDict
    # =============================================================================
    def ComputeGradient(self):
        #        pdb.set_trace()

        for c, value in enumerate(self.MeanMigrationInfoDict):

            GradMeanMigrationInfoArray = np.zeros(
                (self.MeanMigrationInfoDict[value].shape[0], 4)
            )

            GradMeanMigrationInfoArray[:, 2] = np.gradient(
                self.MeanMigrationInfoDict[value][:, 2],
                # self.MeanMigrationInfoDict[value][:,0],
                axis=0,
            )

            GradMeanMigrationInfoArray[:, 0] = self.MeanMigrationInfoDict[value][:, 0]
            GradMeanMigrationInfoArray[:, 1] = self.MeanMigrationInfoDict[value][:, 1]

            self.GradMeanMigrationInfoDict[value] = GradMeanMigrationInfoArray

    @staticmethod
    def MigrationPlotting(FigureNum, MeanMigrationInfoDict, *FileList):
        
#        pdb.set_trace()
        markerList = ["o", "s", "v", "h"]
        lineStyleList = [":", "-.", "--", "-"]
        #        colorList=['#9b1c31','#155b8a','k']
        colorList = ["#9b1c31", "royalblue", "gray", "darkorchid"]
        ax = FigureNum.add_subplot(111)
        for c, value in enumerate(FileList[0]):

            ax.errorbar(
                MeanMigrationInfoDict[value][0:50:3, 0],
                MeanMigrationInfoDict[value][0:50:3, 2],
                yerr=0*MeanMigrationInfoDict[value][0:50:3, 3],
                marker=markerList[c],
                ls=lineStyleList[c],
                color=colorList[c],
                label=FileList[0][c].split("_")[3][0:2] + " " + "mm.s$^{-1}$",
            )
            # FileList[0][c].split('_')[1]+' '+FileList[0][c].split('_')[3]
        #            ax.plot(self.MeanMigrationInfoDict[value][0:70:3,0],
        #                                     self.MeanMigrationInfoDict[value][0:70:3,2],
        #                                     label=FileList[0][c].split('_')[1]+' '+FileList[0][c].split('_')[3])
        #

        ax.set_xlabel(r"Peristalsis cycle", size=6)
        ax.set_ylabel(r"Relative mean displacement (mm)", size=6)
#        ax.set_ylim([-50, 50])
        ax.tick_params(labelsize=6)
        
        ax.legend(prop={"size": 6},
                   borderaxespad=0,
                   handlelength=1,
                   labelspacing=0.3,
                   columnspacing=0.2,
                   loc=0,
                   frameon=False)
        
        #        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        handles, labels = ax.get_legend_handles_labels()
        #        plt.title(FileList[0][c].split('_')[1])
        #        ax.legend.location='east'
        
        #Remove y-ticks
        ax.tick_params(top=False, 
                        bottom=True, 
                        left=True, 
                        right=False, 
                        labelleft=True, 
                        labelbottom=True)
        #Remove borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.show()
        return ax

    @staticmethod
    def MigrationSubPlotting(
        FigureNum, MeanMigrationInfoDict, GradMeanMigrationInfoDict, *FileList
    ):
        #        pdb.set_trace()
        ax = FigureNum.subplots(1, 2)
        #        ax2 = FigureNum.subplots(212,sharex=True)
        for c, value in enumerate(FileList[0]):

            ax[0].errorbar(
                MeanMigrationInfoDict[value][0:50:3, 0],
                MeanMigrationInfoDict[value][0:50:3, 2],
                yerr=MeanMigrationInfoDict[value][0:50:3, 3],
                marker="o",
                ls="--",
                label=FileList[0][c].split("_")[1] + " " + FileList[0][c].split("_")[3],
            )

            ax[1].errorbar(
                GradMeanMigrationInfoDict[value][0:50:3, 0],
                GradMeanMigrationInfoDict[value][0:50:3, 2],
                yerr=GradMeanMigrationInfoDict[value][0:50:3, 3],
                marker="o",
                ls="--",
                label=FileList[0][c].split("_")[1] + " " + FileList[0][c].split("_")[3],
            )
        #

        ax[1].set_xlabel(r"Peristalsis cycle", size=6)
        ax[0].set_ylabel("Relative mean \n displacement (mm)", size=6)
        ax[1].set_ylabel("Gradient mean \n displacement (mm/cycle)", size=6)
        ax[0].legend(prop={"size": 6},
                   borderaxespad=0,
                   handlelength=1,
                   labelspacing=0.1,
                   columnspacing=0.2,
                   loc=0,
                   frameon=False)
        #        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        handles, labels = ax[0].get_legend_handles_labels()
        #Remove y-ticks
        ax[0].tick_params(top=False, 
                        bottom=False, 
                        left=False, 
                        right=False, 
                        labelleft=True, 
                        labelbottom=True)
        #Remove borders
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        
        ax[1].tick_params(top=False, 
                        bottom=False, 
                        left=False, 
                        right=False, 
                        labelleft=True, 
                        labelbottom=True)
        #Remove borders
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        
        plt.show()

    #        ax.legend.location='east'

    @staticmethod
    def MigrationPlottingWithColor(
        FigureNum, MeanMigrationInfoDict, *FileList, **Color
    ):
        #        pdb.set_trace()
        ax = FigureNum.add_subplot(111)
        for c, value in enumerate(FileList[0]):

            ax.errorbar(
                MeanMigrationInfoDict[value][0:50:3, 0],
                MeanMigrationInfoDict[value][0:50:3, 2],
                yerr=MeanMigrationInfoDict[value][0:50:3, 3],
                marker="o",
                ls="--",
                label=FileList[0][c].split("_")[1] + " " + FileList[0][c].split("_")[3],
                color=Color["Color"][c],
            )

        #            ax.plot(self.MeanMigrationInfoDict[value][0:70:3,0],
        #                                     self.MeanMigrationInfoDict[value][0:70:3,2],
        #                                     label=FileList[0][c].split('_')[1]+' '+FileList[0][c].split('_')[3])
        #

        ax.set_xlabel(r"Peristalsis cycle", size=6)
        ax.set_ylim([-50, 50])
        ax.set_ylabel(r"Relative mean displacement (mm)", size=6)
        ax.legend(frameon=False)

        #        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        handles, labels = ax.get_legend_handles_labels()
        #        ax.legend.location='east'
        #Remove y-ticks
        ax.tick_params(top=False, 
                        bottom=False, 
                        left=False, 
                        right=False, 
                        labelleft=True, 
                        labelbottom=True)
        #Remove borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()
        
        
