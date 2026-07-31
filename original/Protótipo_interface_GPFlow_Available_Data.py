# =============================================================================
# IMPORTS
# =============================================================================

# General
import os

# Specific
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkscrolledframe import ScrolledFrame
import ctypes
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colormaps
import matplotlib.colors as mcolors
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gpflow
import pickle
import tempfile
import atexit
from scipy.stats import norm

# =============================================================================
# CONFIGURATIONS
# =============================================================================

# Temporary File creation
temp = tempfile.NamedTemporaryFile(mode="wb", delete=False)
Temp_path = temp.name
temp.close()

atexit.register(lambda: os.remove(Temp_path) if os.path.exists(Temp_path) else None) 

# Appearance
bg_base_color = "gray20"
fg_base_color = "snow"
background_color = "gray10"
entr_btn_color = "gray14"
trainpoint_color = "red"
testpoint_color = "blue"
graph_color = "purple"
font = ("calibri", 10)
font_btn = ("calibri", 8)

# Inicial conditions
type_data = "Manual"
train_done = False
with open(Temp_path, "wb") as file:
        pickle.dump({"type_data": type_data, "train_done": train_done}, file)  

# Pyplot Configuration
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Calibri'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['font.size'] = 8
plt.rcParams["savefig.pad_inches"] = 0.1

# Customization of the utility toolbar
class CustomToolbar(tkagg.NavigationToolbar2Tk):
    toolitems = (('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(background = "White")
        for child in self.winfo_children():
            child.config(background = "White")
            
        self.set_message("(x, y) = (0.0000, 0.0000)")
        if hasattr(self, '_message_label'):
            self._message_label.config(width = 32)
            
    def set_message(self, s):
        if s.strip() != "":
            super().set_message(s)

# Turns the app DPI aware (adapting to the screen's resolution)
# try:
#    ctypes.windll.shcore.SetProcessDpiAwareness(1)
# except Exception:
#     pass

# =============================================================================
# FUNCTIONS
# =============================================================================

# Window ON/OFF Functions======================================================
def Hide_window(window):
    window.withdraw()

def Open_window(window):
    window.deiconify()
    window.lift()

# Important Functions==========================================================
def Update_pickle(path, new_data):
    # Loads old data if it exists
    if os.path.exists(path):
        with open(path, "rb") as file:
            try:
                variables = pickle.load(file)
            except EOFError:
                variables = {}
    else:
        variables = {}

    # Updates the dictionary
    variables.update(new_data)

    # Saves everything again
    with open(path, "wb") as file:
        pickle.dump(variables, file)
        
# DÚVIDA NA DESNORMALIZAÇÃO DE VAR COM LOG+BSTAND
def Normalization(inpt, option,  parms = None, reverse = False,
                  var = {"bol": False, "Y_N": None}):
    """
    Normalization() normalizes the input based on the desired method.

    Parameters
    ----------
    inpt : numpy.array
        Input array of size NxF.
        
    option : string
        Type of normalization. One of:
            - None
            - Standardization
            - LogStand
            - Log + bStand
            - MinMax
            
    parms : numpy.array
        Array of parameters from the input array.
    
    reverse : boolean, optional
        Whether to normalize (False) or unnormalize (True).
        The default is False.
    
    var : boolean, optional
        Whether the input to normalize is a variance (True) or not (False).
        In this work the variance is never normalized, so this function only
        unnormalizes this parameter.
        The default is False.

    Returns
    -------
    outpt : numpy.array
        Output array of size NxF of normalized or unnormalized input
    
    parms : numpy.array
        Array of parameters from the input array.

    """
    if not var["bol"]:
        if option == "None":
            if not reverse:
                if parms is None:
                     mean = inpt.mean(axis = 0)
                     std = inpt.std(axis = 0)
                     parms = [mean, std]
                else:
                   parms = parms
                outpt = inpt
            elif reverse:
                outpt = inpt
                
        elif option == "Standardization":
            if not reverse:
                if parms is None:
                     mean = inpt.mean(axis = 0)
                     std = inpt.std(axis = 0)
                     parms = [mean, std]
                else:
                   parms = parms
                outpt = (inpt - parms[0]) / parms[1]
            elif reverse:
                outpt = (inpt * parms[1]) + parms[0]
                
        elif option == "LogStand":
            if not reverse:
                if parms is None:
                     mean = np.log(inpt).mean(axis = 0)
                     std = np.log(inpt).std(axis = 0)
                     parms = [mean, std]
                else:
                   parms = parms
                outpt = (np.log(inpt)-parms[0]) / parms[1]
            elif reverse:
                outpt = np.exp(inpt * parms[1] + parms[0])
        
        elif option == "Log + bStand":
            b = 10**-3
            if not reverse:
                if parms is None:
                     mean = np.log(inpt + b).mean(axis = 0)
                     std = np.log(inpt + b).std(axis = 0)
                     parms = [mean, std]
                else:
                   parms = parms
                outpt = (np.log(inpt + b)-parms[0]) / parms[1]
            elif reverse:
                outpt = np.exp(inpt * parms[1] + parms[0]) - b
        
        elif option == "MinMax":
            if not reverse:
                if parms is None:
                    inpt_min = np.max(inpt)
                    inpt_max = np.min(inpt)
                    parms = [inpt_min, inpt_max]
                else:
                    parms = parms
                outpt = (inpt-parms[0])/(parms[1]-parms[0])
            elif reverse:
                outpt = inpt * (parms[1]-parms[0]) + parms[0]
    elif var["bol"]:
        if option == "None":
            if reverse:
                outpt = inpt
                
        elif option == "Standardization":
            if reverse:
                outpt = inpt * parms[1]**2
        
        elif option == "LogStand":
            if reverse:
                outpt = inpt * (parms[1] * np.exp(parms[0] + parms[1] * var["Y_N"])) ** 2
        
        elif option == "Log + bStand":
            if reverse:
                outpt = inpt * (parms[1] * np.exp(parms[0] + parms[1] * var["Y_N"])) ** 2
        
        elif option == "MinMax":
            if reverse:
                outpt = inpt * (parms[1] - parms[0]) ** 2
                
    return outpt, parms

# Model Training/Ploting=======================================================
def Train():
    global Y_Train,Y_Test,X_Train,X_Test,X,Y
    # Gets the type of data from the temporary pickle file
    with open(Temp_path, "rb") as file:
            variables = pickle.load(file)
            type_data = variables["type_data"]
    
    # Type of normalization in use
    var_norm_label = str(selected_norm_label.get())
    var_norm_feat = str(selected_norm_feat.get())
    var_kernel = str(selected_kernel.get())
    
    # Checks if trainlikelihood is active or not
    if checkvar1.get() == 1:
        trainlikelihood = True
    elif checkvar1.get() == 0:
        trainlikelihood = False
    
    # Types of kernel
    if var_kernel == "RBF":
        kernel_in_use = gpflow.kernels.SquaredExponential()
    elif var_kernel == "Matern 12":
        kernel_in_use = gpflow.kernels.Matern12()
    elif var_kernel == "Matern 32":
        kernel_in_use = gpflow.kernels.Matern32()
    elif var_kernel == "Matern 52":
        kernel_in_use = gpflow.kernels.Matern52()
    elif var_kernel == "Periodic":
        kernel_in_use = gpflow.kernels.Periodic(
            gpflow.kernels.SquaredExponential())
    elif var_kernel == "RQ":
        kernel_in_use = gpflow.kernels.RationalQuadratic()
    
    # Checks if white kernel is active or not
    if checkvar7.get() == 1:
        white_kernel = True
        kernel_in_use = kernel_in_use + gpflow.kernels.White()
    elif checkvar7.get() == 0:
        white_kernel = False
        kernel_in_use = kernel_in_use
        
    # Dataset treatment for manual type data
    if type_data == "Manual":
        X = np.array([float(e.get()) for e in inpt1]).reshape((-1,1))
        Y = np.array([float(e.get()) for e in inpt2]).reshape((-1,1))
        ncol_data = 2
        
        Axes_titles = list(["X", "Y"])
        Graph_title = "Manual Points Graph (for testing porposes)"
        
    # Dataset treatment for import type data
    elif type_data == "Import":
        with open(Temp_path, "rb") as file:
                variables = pickle.load(file)
                file_path = variables["file_path"]
        
        # Gets the title for plots from the first line of the CSV file
        Comments_csv = []
        with open(file_path, 'r', encoding="utf-8-sig") as fobj:
            for line in fobj:
                if line.strip().startswith("#"):
                    Comments_csv.append(line.strip())
                else:
                    break    
        Graph_title = Comments_csv[0][1:].strip(",")
        
        # Enables the variable selection window and imports from the pickle
        # file the new dataset
        Select_var(file_path)
        with open(Temp_path, "rb") as file:
                variables = pickle.load(file)
                df_new = variables["df_new"]
                    
        # Collumn names/Axes titles for plot
        Col_names = list(df_new)
        Axes_titles = [n[:10] + "..." if len(n) > 10 else n for n in Col_names]
        
        # Division of the new dataset into features (X) and label (Y)
        ncol_data = df_new.shape[1]
        X = df_new.iloc[:,:-1].values[:, None].reshape(-1, ncol_data-1)
        Y = df_new.iloc[:,-1].values[:, None]
        
        X = X.astype(float)
        Y = Y.astype(float)
    
    # Train/Test split
    if checkvar2.get() == 1:
        Split_Percentage = float(Test_Percentage.get())
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,
                                                        random_state = 11,
                                                        test_size = Split_Percentage/100)
        Train_Test_Split = True
    else:
        Split_Percentage = 0
        X_Train = X
        Y_Train = Y
        X_Test = []
        Y_Test = []
        Train_Test_Split = False

    # Normalized train points
    X_Train_N, parms_X = Normalization(X_Train, var_norm_feat, 
                                       parms = None, reverse = False)
    Y_Train_N, parms_Y = Normalization(Y_Train, var_norm_label,
                                       parms = None, reverse = False)
    
    # Creation and optimization of a model with the normalized points
    model = gpflow.models.GPR((X_Train_N, Y_Train_N), kernel = kernel_in_use,
                              noise_variance = 10**-5)
    
    gpflow.utilities.set_trainable(model.likelihood.variance, trainlikelihood)
    
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    
    # Number of hyperparameters used in the model
    num_params = sum(p.shape.num_elements() for p in model.trainable_parameters)
    
    # Tells function "Plot_graph" that the train is done
    train_done = True
    
    # Disables widgets no longer necessary
    for child in Frame_kernel_norm.winfo_children():
        child.configure(state = 'disabled')
    
    for child in Frame_split.winfo_children():
        child.configure(state = 'disabled')
        
    bt2.configure(state = "disabled")
    
    for i in range(rows):
        for j in range(cols):
            entry = entries[i][j]
            entry.config(state = "disabled")
    
    bt1.configure(state = "disabled")
    
    # Destroys and clears the old "Predict Y Window" entries
    for i in range(0,len(x_value_entry)):
        x_value_entry[i].destroy()
        lbl_Pred_Titles[i].destroy()
    x_value_entry.clear()
    lbl_Pred_Titles.clear()
    
    # Creates new "Predict Y Window" entries
    for i in range(0,ncol_data-1):
        x_value_entry.append(tk.Entry(inner_frame_Pred, width = 12, bg = entr_btn_color,
                              fg = fg_base_color,
                              insertbackground = fg_base_color))
        x_value_entry[i].grid(row = 1, column = i, padx=5, pady=2)
        
        lbl_Pred_Titles.append(tk.Label(inner_frame_Pred, text = "", font = font,
                             bg = bg_base_color, fg = fg_base_color))
        lbl_Pred_Titles[i].grid(row = 0, column = i, padx=5, pady=2)
    
    # Destroys and clears the old "AL and BO Window" entries
    for i in range(0,len(x_min_entry_ALBO)):
        x_min_entry_ALBO[i].destroy()
        x_max_entry_ALBO[i].destroy()
        lbl_ALBO_Titles[i].destroy()
        lbl_ALBO_Results[i].destroy()
    x_min_entry_ALBO.clear()
    x_max_entry_ALBO.clear()
    lbl_ALBO_Titles.clear()
    lbl_ALBO_Results.clear()

    # Creates new "Bayesian Optimization Window" entries
    for i in range(0,ncol_data-1):
        x_min_entry_ALBO.append(tk.Entry(inner_frame_ALBO, width = 10, bg = entr_btn_color,
                              fg = fg_base_color,
                              insertbackground = fg_base_color))
        x_min_entry_ALBO[i].grid(row = i, column = 1, padx=5, pady=2)
        x_max_entry_ALBO.append(tk.Entry(inner_frame_ALBO, width = 10, bg = entr_btn_color,
                              fg = fg_base_color,
                              insertbackground = fg_base_color))
        x_max_entry_ALBO[i].grid(row = i, column = 2, padx=5, pady=2)
        
        lbl_ALBO_Titles.append(tk.Label(inner_frame_ALBO, text = "", font = font,
                             bg = bg_base_color, fg = fg_base_color))
        lbl_ALBO_Titles[i].grid(row = i, column = 0, padx=5, pady=2, sticky="w")

        lbl_ALBO_Results.append(tk.Label(inner_frame_ALBO, text = "", font = font,
                             bg = bg_base_color, fg = fg_base_color))
        lbl_ALBO_Results[i].grid(row = i, column = 3, padx=5, pady=2, sticky="w")
    
    if ncol_data == 2:
        entry_plot_min[0].configure(state = "normal")
        entry_plot_max[0].configure(state = "normal")
        lbl_Plot_Titles[0].configure(text = Axes_titles[0] + " :")
    elif ncol_data == 3:
        for i in range(0,2):
            entry_plot_min[i].configure(state = "normal")
            entry_plot_max[i].configure(state = "normal")
            lbl_Plot_Titles[i].configure(text = Axes_titles[i] + " :")
    elif ncol_data > 3 or ncol_data == 1:
        for i in range(0,2):
            entry_plot_min[i].configure(state = "disabled")
            entry_plot_max[i].configure(state = "disabled")
            lbl_Plot_Titles[i].configure(text = "Variable" + str(i+1) + ":")
    
    # Updates "Frame_info" with the new model information
    lbl15.configure(text = ncol_data)
    lbl16.configure(text = np.size(X_Train,0))
    lbl17.configure(text = np.size(X_Test,0))
    lbl18.configure(text = num_params)
    
    # Updates Predict Y Window labels
    for i in range(ncol_data - 1):
        lbl_Pred_Titles[i].configure(text = Axes_titles[i])
    
    lbl2_Pred.configure(text = "Pred. " + Axes_titles[-1] + " :")
    
    # Updates AL and BO Window labels
    for i in range(ncol_data - 1):
        lbl_ALBO_Titles[i].configure(text = Axes_titles[i] + " :")
    
    # Dumbs variables into temporary pickle file
    Update_pickle(Temp_path,{"model": model, "parms_X": parms_X,
                             "parms_Y": parms_Y, "train_done": train_done,
                             "X_Train": X_Train,"Y_Train": Y_Train,
                             "X_Test": X_Test, "Y_Test": Y_Test,
                             "ncol_data": ncol_data, "X": X, "Y": Y,
                             "Axes_titles": Axes_titles,
                             "Graph_title": Graph_title,
                             "var_norm_label": var_norm_label,
                             "var_norm_feat": var_norm_feat,
                             "var_kernel": var_kernel,
                             "trainlikelihood": trainlikelihood,
                             "white_kernel": white_kernel,
                             "Train_Test_Split": Train_Test_Split,
                             "Split_Percentage": Split_Percentage,
                             "num_params": num_params})

def Plot_parity():
    global utilitybar
    with open(Temp_path, "rb") as file:
        variables = pickle.load(file)
        train_done = variables["train_done"]
        
    figure = plt.figure(figsize = (4.5,2.5), dpi = 100)
    figure.add_subplot(111).plot()
    plt.grid()
    plt.title("Parity plot")
    plt.xlabel("Exp. Y")
    plt.ylabel("Pred. Y")
    plt.tight_layout()
    chart = FigureCanvasTkAgg(figure, master = Frame_parity)
    chart.get_tk_widget().place(x = 10, y = 10)
    plt.close()

    utilitybar = CustomToolbar(chart, Frame_parity) # Inicialize utility bar
    utilitybar.place(x = 10, y = 260)
    
    if train_done:
        with open(Temp_path, "rb") as file:
            variables = pickle.load(file)
            Axes_titles = variables["Axes_titles"]
            model = variables["model"]
            parms_X = variables["parms_X"]
            parms_Y = variables["parms_Y"]
            X_Train = variables["X_Train"]
            Y_Train = variables["Y_Train"]
            X_Test = variables["X_Test"]
            Y_Test = variables["Y_Test"]
            Y = variables["Y"]
            var_norm_label = variables["var_norm_label"]
            var_norm_feat = variables["var_norm_feat"]
        
        X_Train_N, _= Normalization(X_Train, var_norm_feat, 
                                    parms = None, reverse = False)
        if checkvar2.get() == 1:
            X_Test_N, _ = Normalization(X_Test, var_norm_feat,
                                        parms = parms_X, reverse = False)
        
        # Elimination of the old utility bar
        utilitybar.destroy()
        
        # Parity plot
        Y_Train_pred_N, Y_Train_pred_std_N = model.predict_y(X_Train_N, full_cov = False)
        
        Y_Train_pred = Normalization(np.array([Y_Train_pred_N]).reshape(-1,1),
                               var_norm_label, parms = parms_Y, reverse = True)
        Y_Train_pred_std = Normalization(np.array([Y_Train_pred_std_N]).reshape(-1,1),
                                         var_norm_label, parms = parms_Y,
                                         reverse = True,
                                         var = {"bol": True, "Y_N": Y_Train_pred_N})
        
        if checkvar2.get() == 1:
            Y_Test_pred_N, Y_Test_pred_std_N = model.predict_y(X_Test_N, full_cov = False)
            Y_Test_pred = Normalization(np.array([Y_Test_pred_N]).reshape(-1,1),
                                   var_norm_label, parms = parms_Y, reverse = True)
            Y_Test_pred_std = Normalization(np.array([Y_Test_pred_std_N]).reshape(-1,1),
                                             var_norm_label, parms = parms_Y,
                                             reverse = True,
                                             var = {"bol": True, "Y_N": Y_Test_pred_N})
            
        # R2, MAE, MAPE, RMSE
        R2_Train = metrics.r2_score(Y_Train, Y_Train_pred[0])
        RMSE_Train = metrics.root_mean_squared_error(Y_Train, Y_Train_pred[0])
        MAE_Train = metrics.mean_absolute_error(Y_Train, Y_Train_pred[0])
        MAPE_Train = metrics.mean_absolute_percentage_error(Y_Train, Y_Train_pred[0])
        
        if checkvar2.get() == 1:
            R2_Test = metrics.r2_score(Y_Test, Y_Test_pred[0])
            RMSE_Test = metrics.root_mean_squared_error(Y_Test, Y_Test_pred[0])
            MAE_Test = metrics.mean_absolute_error(Y_Test, Y_Test_pred[0])
            MAPE_Test = metrics.mean_absolute_percentage_error(Y_Test, Y_Test_pred[0])
            
        figure3 = plt.figure(figsize = (4.5,2.5), dpi = 100)
        
        ax = figure3.add_subplot(111)
        ax.scatter(Y_Train, Y_Train_pred[0], s = 20, facecolors = 'none',
                       edgecolors = 'red', label = "Train")
        if checkvar2.get() == 1:
            ax.scatter(Y_Test, Y_Test_pred[0], color = "blue", s = 20,
                       marker = "x", label = "Test")
        ax.plot((min(Y),max(Y)), (min(Y),max(Y)),color='k', linestyle = '--',
                linewidth = 1)
        plt.legend(fontsize = 8)
        
        text_position = 0.93
        
        if checkvar3.get() == 1:
            plt.text(0.02, text_position, 'MAE (Train) = ' + '{:.3f} '.format(MAE_Train),
                 horizontalalignment='left',
                 transform=plt.gca().transAxes, c='r')
            text_position = text_position - 0.08
        if checkvar4.get() == 1:
            plt.text(0.02, text_position, 'MAPE (Train) = ' + '{:.3f} '.format(MAPE_Train),
                 horizontalalignment='left',
                 transform=plt.gca().transAxes, c='r')
            text_position = text_position - 0.08
        if checkvar5.get() == 1:    
            plt.text(0.02, text_position, 'R² (Train) = ' + '{:.3f} '.format(R2_Train),
                 horizontalalignment='left',
                 transform=plt.gca().transAxes, c='r')
            text_position = text_position - 0.08
        if checkvar6.get() == 1:
            plt.text(0.02, text_position, 'RMSE (Train) = ' + '{:.3f} '.format(RMSE_Train),
                 horizontalalignment='left',
                 transform=plt.gca().transAxes, c='r')
            text_position = text_position - 0.08
        
        if checkvar2.get() == 1:
            if checkvar3.get() == 1:
                plt.text(0.02, text_position, 'MAE (Test) = ' + '{:.3f} '.format(MAE_Test),
                     horizontalalignment='left',
                     transform=plt.gca().transAxes, c = 'b')
                text_position = text_position - 0.08
            if checkvar4.get() == 1:
                plt.text(0.02, text_position, 'MAPE (Test) = ' + '{:.3f} '.format(MAPE_Test),
                     horizontalalignment='left',
                     transform=plt.gca().transAxes, c='b')
                text_position = text_position - 0.08
            if checkvar5.get() == 1:
                plt.text(0.02, text_position, 'R² (Test) = ' + '{:.3f} '.format(R2_Test),
                     horizontalalignment='left',
                     transform=plt.gca().transAxes, c='b')
                text_position = text_position - 0.08
            if checkvar6.get() == 1:
                plt.text(0.02, text_position, 'RMSE (Test) = ' + '{:.3f} '.format(RMSE_Test),
                     horizontalalignment='left',
                     transform=plt.gca().transAxes, c='b')
                text_position = text_position - 0.08
        
        if checkvar8.get() == 1:
            plt.errorbar(Y_Train.reshape(-1,), Y_Train_pred[0].reshape(-1,),
                         yerr = np.sqrt(Y_Train_pred_std[0]).reshape(-1,),
                         fmt='o', linestyle = 'none', capsize = 3,
                         color = 'black', ecolor = 'black', markersize = 1)
            if checkvar2.get() == 1:
                plt.errorbar(Y_Test.reshape(-1,), Y_Test_pred[0].reshape(-1,),
                             yerr = np.sqrt(Y_Test_pred_std[0]).reshape(-1,),
                             fmt='o', linestyle = 'none', capsize = 3,
                             color = 'black', ecolor = 'black', markersize = 1)
        
        plt.title("Parity plot")
        plt.xlabel("Exp. " + Axes_titles[-1])
        plt.ylabel("Pred. " + Axes_titles[-1])
        plt.tight_layout()
        
        chart = FigureCanvasTkAgg(figure3, master = Frame_parity)
            
        #plt.grid()
        chart.get_tk_widget().place(x = 10, y = 10)
        plt.close()
        
        # Creation of a new utility bar
        utilitybar = CustomToolbar(chart, Frame_parity)
        utilitybar.place(x = 10, y = 260)
        utilitybar.update()

# ERROR BARS
def Plot_graph():
    global utilitybar, X_Plot, Plot_min
    with open(Temp_path, "rb") as file:
        variables = pickle.load(file)
        train_done = variables["train_done"]
    
    # Empty plot
    figure = plt.figure(figsize = (4.5,3), dpi = 100)
    figure.add_subplot(111).plot()
    plt.title("GRAPH")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.tight_layout()
    chart = FigureCanvasTkAgg(figure, master = Frame_plot)
    plt.close()
    
    utilitybar = CustomToolbar(chart, Frame_plot)
    utilitybar.place(x = 10, y = 310)

    if train_done:
        with open(Temp_path, "rb") as file:
            variables = pickle.load(file)
            Axes_titles = variables["Axes_titles"]
            Graph_title = variables["Graph_title"]
            model = variables["model"]
            parms_X = variables["parms_X"]
            parms_Y = variables["parms_Y"]
            X_Train = variables["X_Train"]
            Y_Train = variables["Y_Train"]
            X_Test = variables["X_Test"]
            Y_Test = variables["Y_Test"]
            ncol_data = variables["ncol_data"]
            var_norm_label = variables["var_norm_label"]
            var_norm_feat = variables["var_norm_feat"]
        
        N_Points = round(int(selected_N_points.get()) ** (1 / (ncol_data - 1)))
        X_Plot = np.zeros((N_Points, ncol_data - 1))
        
        if var_sel_plot.get() == 1:
            with open(Temp_path, "rb") as file:
                    variables = pickle.load(file)
                    X = variables["X"]
                    
            for n in range(ncol_data - 1):
                varRange = np.linspace(X[:,n].min(), X[:,n].max(), N_Points)
                X_Plot[:,n] = varRange.copy()
        else:
            if ncol_data == 2:
                Plot_min = np.array([float(entry_plot_min[0].get())]).reshape(-1, 1)
                Plot_max = np.array([float(entry_plot_max[0].get())]).reshape(-1, 1)
            else:
                Plot_min = np.array([float(e.get()) for e in entry_plot_min]).reshape(-1, 1)
                Plot_max = np.array([float(e.get()) for e in entry_plot_max]).reshape(-1, 1)
            Plot_Limits = np.hstack((Plot_min, Plot_max))
            
            for n in range(ncol_data - 1):
                varRange = np.linspace(Plot_Limits[n,:].min(), Plot_Limits[n,:].max(), N_Points)
                X_Plot[:,n] = varRange.copy()
                
        X_Plot = np.array(np.meshgrid(*[X_Plot[:, i] for i in range(ncol_data - 1)]))\
            .T.reshape(-1, ncol_data - 1)
        
        utilitybar.destroy()
        
        X_Plot_N = Normalization(X_Plot, var_norm_feat, parms = parms_X,
                                 reverse = False)
        Y_mean_N, Y_var_N = model.predict_y(X_Plot_N[0], full_cov = False)
        
        # Unnormalized Y_mean
        Y_mean = Normalization(np.array([Y_mean_N]).reshape(-1,1),
                               var_norm_label, parms = parms_Y, reverse = True)
        Y_var = Normalization(np.array([Y_var_N]).reshape(-1,1), var_norm_label,
                              parms = parms_Y, reverse = True,
                              var = {"bol": True, "Y_N": Y_mean_N})
        
        if ncol_data == 2:
            # 95% confidence interval for Y_mean
            Y_Upper = Y_mean[0] + 1.96 * np.sqrt(Y_var[0])
            Y_Lower = Y_mean[0] - 1.96 * np.sqrt(Y_var[0])
            
            figure1 = plt.figure(figsize = (4.5,3), dpi = 100)
            
            ax = figure1.add_subplot(111)
    
            ax.plot(X_Plot, Y_mean[0], label = "Y mean", color = selected_model_color.get())
            ax.plot(X_Plot, Y_Upper, "--", label = "Y I.C. 95%", color = selected_IC_color.get())
            ax.plot(X_Plot, Y_Lower, "--", color = selected_IC_color.get())
            ax.plot(X_Train, Y_Train, "o", label = "Train", color = trainpoint_color)
            
            if checkvar2.get() == 1:
                ax.plot(X_Test, Y_Test, "o", label = "Test", color = testpoint_color)

            ax.legend(fontsize = 8)
            plt.fill_between(X_Plot[:,0], Y_Lower[:,0], Y_Upper[:,0],
                             color = selected_IC_color.get(), alpha = 0.1)
            plt.title(Graph_title)
            plt.xlabel(Axes_titles[0])
            plt.ylabel(Axes_titles[1])
            plt.tight_layout()
            
            chart = FigureCanvasTkAgg(figure1, master = Frame_plot)
            
        elif ncol_data == 3:
            cmap_colour = str(selected_cmap.get())
            figure2 = plt.figure(figsize = (4.5,3), dpi = 100)
            ax = figure2.add_subplot(111, projection='3d', computed_zorder=False)
            #ax.set_proj_type('persp',focal_length=10)
            ax.plot_trisurf(X_Plot[:,0], X_Plot[:,1], Y_mean[0].reshape(-1,),
                                 cmap = cmap_colour)
            ax.view_init(elev = 15, azim = 310)
            
            ax.plot(X_Train[:,0], X_Train[:,1],
                    Y_Train[:,0], "o", color = trainpoint_color, zorder = 4.6, markersize = 3)
            if checkvar2.get() == 1:
                ax.plot(X_Test[:,0], X_Test[:,1],
                        Y_Test[:,0], 'o', color = testpoint_color, zorder = 4.6, markersize = 3)
            
            plt.title(Graph_title)
            ax.set_xlabel(Axes_titles[0])
            ax.set_ylabel(Axes_titles[1])
            ax.set_zlabel(Axes_titles[2])
            plt.tight_layout()

            chart = FigureCanvasTkAgg(figure2, master = Frame_plot)
        
        else:
            figure = plt.figure(figsize = (4.5,3), dpi = 100)
            figure.add_subplot(111).plot()
            plt.title("GRAPH")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.tight_layout()
            chart = FigureCanvasTkAgg(figure, master = Frame_plot)
        
        plt.grid()
        utilitybar = CustomToolbar(chart, Frame_plot)
        utilitybar.place(x = 10, y = 310)
        utilitybar.update()

    chart.get_tk_widget().place(x = 10, y = 10)
    plt.close()

def Plot_AF():
    global utilitybar, X
    with open(Temp_path, "rb") as file:
        variables = pickle.load(file)
        train_done = variables["train_done"]
    
    figure = plt.figure(figsize = (4.5,3), dpi = 100)
    figure.add_subplot(111).plot()
    plt.title("AF GRAPH")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.tight_layout()
    chart = FigureCanvasTkAgg(figure, master = Frame_AF_Plot)
    plt.close()
    
    utilitybar = CustomToolbar(chart, Frame_AF_Plot)
    utilitybar.place(x = 10, y = 310)

    if train_done:
        with open(Temp_path, "rb") as file:
            variables = pickle.load(file)
            Axes_titles = variables["Axes_titles"]
            Graph_title = variables["Graph_title"]
            model = variables["model"]
            parms_X = variables["parms_X"]
            parms_Y = variables["parms_Y"]
            X_Train = variables["X_Train"]
            Y_Train = variables["Y_Train"]
            X_Test = variables["X_Test"]
            Y_Test = variables["Y_Test"]
            ncol_data = variables["ncol_data"]
            var_norm_label = variables["var_norm_label"]
            var_norm_feat = variables["var_norm_feat"]
        
        if ncol_data >3 or ncol_data <2:
            return
        
        N_Points = round(int(selected_N_points_ALBO.get()) ** (1 / (ncol_data - 1)))
        X_Plot = np.zeros((N_Points, ncol_data - 1))
        
        if var_sel_ALBO.get() == 1:
            with open(Temp_path, "rb") as file:
                    variables = pickle.load(file)
                    X = variables["X"]
                    
            for n in range(ncol_data - 1):
                varRange = np.linspace(X[:,n].min(), X[:,n].max(), N_Points)
                X_Plot[:,n] = varRange.copy()
        else:
            if ncol_data == 2:
                Plot_min = np.array([float(x_min_entry_ALBO[0].get())]).reshape(-1, 1)
                Plot_max = np.array([float(x_max_entry_ALBO[0].get())]).reshape(-1, 1)
            elif ncol_data == 3:
                Plot_min = np.array([float(e.get()) for e in x_min_entry_ALBO[0:2]]).reshape(-1, 1)
                Plot_max = np.array([float(e.get()) for e in x_max_entry_ALBO[0:2]]).reshape(-1, 1)
            Plot_Limits = np.hstack((Plot_min, Plot_max))
            
            for n in range(ncol_data - 1):
                varRange = np.linspace(Plot_Limits[n,:].min(), Plot_Limits[n,:].max(), N_Points)
                X_Plot[:,n] = varRange.copy()
                
        X_Plot = np.array(np.meshgrid(*[X_Plot[:, i] for i in range(ncol_data - 1)]))\
            .T.reshape(-1, ncol_data - 1)
        
        utilitybar.destroy()
        
        X_Plot_N = Normalization(X_Plot, var_norm_feat, parms = parms_X,
                                 reverse = False)
        Y_mean_N, Y_var_N = model.predict_y(X_Plot_N[0], full_cov = False)
        
        # Unnormalized Y_mean
        Y_mean = Normalization(np.array([Y_mean_N]).reshape(-1,1),
                               var_norm_label, parms = parms_Y, reverse = True)
        Y_var = Normalization(np.array([Y_var_N]).reshape(-1,1), var_norm_label,
                              parms = parms_Y, reverse = True,
                              var = {"bol": True, "Y_N": Y_mean_N})
        
        var_AF = str(selected_AF.get())
        # Types of Aquisition Function (A.F.)
        if var_AF == "PI":
            AF = norm.cdf((Y_mean[0] - max(Y_mean[0])) / Y_var[0], 0 , 1)
        elif var_AF == "EI":
            AF = (Y_mean[0] - max(Y_mean[0]))\
                * norm.cdf((Y_mean[0] - max(Y_mean[0])) / Y_var[0], 0 , 1) + Y_var[0]\
                    * norm.pdf((Y_mean[0] - max(Y_mean[0])) / Y_var[0], 0 , 1)
        elif var_AF == "UCB":
            lamda = 1
            AF = Y_mean[0] + lamda * Y_var[0]
        elif var_AF == "Std":
            AF = np.sqrt(Y_var[0])
        elif var_AF == "Std/Mean":
            AF = np.sqrt(Y_var[0])/Y_mean[0]
        
        if ncol_data == 2:            
            figure1 = plt.figure(figsize = (4.5,3), dpi = 100)
            
            ax = figure1.add_subplot(111)
    
            ax.plot(X_Plot, Y_mean[0], label = "Y mean", color = selected_model_color_ALBO.get())
            ax.plot(X_Plot, AF, "--", label = "AF", color = selected_AF_color.get())
            ax.plot(X_Train, Y_Train, "o", label = "Train", color = trainpoint_color)
            
            if checkvar2.get() == 1:
                ax.plot(X_Test, Y_Test, "o", label = "Test", color = testpoint_color)

            ax.legend(fontsize = 8)
            plt.title(Graph_title)
            plt.xlabel(Axes_titles[0])
            plt.ylabel("A.F.")
            plt.tight_layout()
            
            chart = FigureCanvasTkAgg(figure1, master = Frame_AF_Plot)
            
        elif ncol_data == 3:
            cmap_colour = str(selected_ALBO_cmap.get())
            figure2 = plt.figure(figsize = (4.5,3), dpi = 100)
            ax = figure2.add_subplot(111, projection='3d', computed_zorder=False)
            #ax.set_proj_type('persp',focal_length=10)
            ax.plot_trisurf(X_Plot[:,0], X_Plot[:,1], AF.reshape(-1,),
                                 cmap = cmap_colour)
            ax.view_init(elev = 15, azim = 310)
            
            ax.plot(X_Train[:,0], X_Train[:,1],
                    Y_Train[:,0], "o", color = trainpoint_color, zorder = 4.6, markersize = 3)
            if checkvar2.get() == 1:
                ax.plot(X_Test[:,0], X_Test[:,1],
                        Y_Test[:,0], 'o', color = testpoint_color, zorder = 4.6, markersize = 3)
            
            plt.title(Graph_title)
            ax.set_xlabel(Axes_titles[0])
            ax.set_ylabel(Axes_titles[1])
            ax.set_zlabel("A.F.")
            plt.tight_layout()

            chart = FigureCanvasTkAgg(figure2, master = Frame_AF_Plot)
        
        else:
            figure = plt.figure(figsize = (4.5,3), dpi = 100)
            figure.add_subplot(111).plot()
            plt.title("GRAPH")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.tight_layout()
            chart = FigureCanvasTkAgg(figure, master = Frame_AF_Plot)
        
        plt.grid()
        utilitybar = CustomToolbar(chart, Frame_AF_Plot)
        utilitybar.place(x = 10, y = 310)
        utilitybar.update()

    chart.get_tk_widget().place(x = 10, y = 10)
    plt.close()

# Other Functions==============================================================
def Open_CSV_File():
    file_path = filedialog.askopenfilename(title = "Open CSV File", filetypes = [("CSV files", "*.csv")])
    
    if len(file_path) != 0:
        for i in range(rows):
            for j in range(cols):
                entry = entries[i][j]
                entry.config(state = "disabled")
        type_data = "Import"
    else:
        type_data = "Manual"
    
    Update_pickle(Temp_path, {"type_data": type_data, "file_path": file_path})

def Clean_graphs():
    Plot_graph()
    Plot_AF()
    Plot_parity()

def Global_Reset():   
    type_data = "Manual"
    train_done = False
    
    for i in range(rows):
        for j in range(cols):
            entry = entries[i][j]
            entry.config(state = "normal")
            
    for child in Frame_kernel_norm.winfo_children():
        if child.winfo_class() == "TCombobox":
            child.configure(state = 'readonly')
        else:
            child.configure(state = 'normal')
    
    for child in Frame_split.winfo_children():
        child.configure(state = 'normal')
        
    if checkvar2.get() == 0:
        Test_Percentage.configure(state = "disabled")
    
    # Destroys and clears the old "Predict Y Window" entries and labels
    for i in range(0,len(x_value_entry)):
        x_value_entry[i].destroy()
        lbl_Pred_Titles[i].destroy()
    x_value_entry.clear()
    lbl_Pred_Titles.clear()
    
    # Creates new "Predict Y Window" entries
    x_value_entry.append(tk.Entry(inner_frame_Pred, width = 10, bg = entr_btn_color,
                                  font = font, fg = fg_base_color,
                                  insertbackground = fg_base_color, state = "disabled"))
    x_value_entry[0].grid(row = 1, column = 0, padx=5, pady=2)
    
    # Destroys and clears the old "AL and BO Window" entries
    for i in range(0,len(x_min_entry_ALBO)):
        x_min_entry_ALBO[i].destroy()
        x_max_entry_ALBO[i].destroy()
        lbl_ALBO_Titles[i].destroy()
        lbl_ALBO_Results[i].destroy()
    x_min_entry_ALBO.clear()
    x_max_entry_ALBO.clear()
    lbl_ALBO_Titles.clear()
    lbl_ALBO_Results.clear()

    # Creates new "Bayesian Optimization Window" entries
    x_min_entry_ALBO.append(tk.Entry(inner_frame_ALBO, width = 8, bg = entr_btn_color,
                                  font = font, fg = fg_base_color,
                                  insertbackground = fg_base_color, state = "disabled"))
    x_min_entry_ALBO[0].grid(row = 0, column = 1, padx=5, pady=2)
    
    x_max_entry_ALBO.append(tk.Entry(inner_frame_ALBO, width = 8, bg = entr_btn_color,
                                  font = font, fg = fg_base_color,
                                  insertbackground = fg_base_color, state = "disabled"))
    x_max_entry_ALBO[0].grid(row = 0, column = 2, padx=5, pady=2)
    
    for i in range(0,2):
        entry_plot_min[i].configure(state = "disabled")
        entry_plot_max[i].configure(state = "disabled")
        lbl_Plot_Titles[i].configure(text = "Variable" + str(i+1) + ":")
    
    bt1.configure(state = "normal")
    bt2.configure(state = "normal")
    bt5.configure(state = "normal")

    # Updates "Frame_info" lables
    lbl15.configure(text = " --- ")    
    lbl16.configure(text = " --- ")    
    lbl17.configure(text = " --- ")    
    lbl18.configure(text = " --- ")    
    
    # Updates Predict Y Window labels
    lbl_Pred_Titles.append(tk.Label(inner_frame_Pred, text = "Variable 1", font = font,
                         bg = bg_base_color, fg = fg_base_color))
    lbl_Pred_Titles[0].grid(row = 0, column = 0, padx=5, pady=2)
    
    lbl2_Pred.configure(text = "Pred. Y :")
    
    # Updates AL and BO Window labels
    lbl_ALBO_Titles.append(tk.Label(inner_frame_ALBO, text = "Variable 1 :", font = font,
                         bg = bg_base_color, fg = fg_base_color))
    lbl_ALBO_Titles[0].grid(row = 0, column = 0, padx=5, pady=2, sticky="w")

    lbl_ALBO_Results.append(tk.Label(inner_frame_ALBO, text = "", font = font,
                         bg = bg_base_color, fg = fg_base_color))
    lbl_ALBO_Results[0].grid(row = 0, column = 3, padx=5, pady=2, sticky="w")
    
    with open(Temp_path, "wb") as file:
        pickle.dump({}, file)
    
    Update_pickle(Temp_path, {"type_data": type_data, "train_done": train_done})
        
    Clean_graphs()

def Config_Reset():
    for child in Frame_kernel_norm.winfo_children():
        if child.winfo_class() == "TCombobox":
            child.configure(state = 'readonly')
        else:
            child.configure(state = 'normal')
    
    for child in Frame_split.winfo_children():
        child.configure(state = 'normal')
        
    if checkvar2.get() == 0:
        Test_Percentage.configure(state = "disabled")
    
    bt1.configure(state = "normal")

def Do_Split():
    if checkvar2.get() == 1:
        Test_Percentage.configure(state = "normal")
    elif checkvar2.get() == 0:
        Test_Percentage.configure(state = "disabled")

def move_focus(entries, row, col, d_row, d_col):
    new_row = row + d_row
    new_col = col + d_col
    
    # Limitar dentro dos índices válidos
    new_row = max(0, min(new_row, len(entries) - 1))
    new_col = max(0, min(new_col, len(entries[0]) - 1))
    
    entries[new_row][new_col].focus_set()

def find_widgets_by_type(root, cls):
    found = []
    for child in root.winfo_children():
        if isinstance(child, cls):
            found.append(child)
        found.extend(find_widgets_by_type(child, cls))
    return found

def Select_var(file_path):
   
    with open(file_path, 'r') as f:
    # Saves the number of the line that begins with "#"
        comment_lines = [i for i, line in enumerate(f) if line.strip().startswith('#')]

    df = pd.read_csv(file_path, skiprows=comment_lines)
    df_columns = df.columns.tolist()
    
    root_col = tk.Toplevel(root)
    root_col.title("Select desired data")
    root_col.geometry("225x220")
    root_col.resizable(False, False)
    root_col.configure(borderwidth = 10, bg = background_color)
    
    Frame_col = tk.Frame(root_col, width = 205, height = 200, bg = bg_base_color)
    Frame_col.place(x = 0, y = 0)
    
    lbl_label = tk.Label(Frame_col, text = "Label: ", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
    lbl_label.place(x = 10, y = 10)
    
    Frame_Scroll_col = tk.Frame(root_col, width = 205, height = 120, bg = bg_base_color)
    Frame_Scroll_col.place(x = 0, y = 40)

    sf_col = ScrolledFrame(Frame_Scroll_col, width = 185, height = 120,
                            scrollbars = "vertical", use_ttk = False)
    sf_col.pack(fill = "both", expand = True)
    
    canvas_col = find_widgets_by_type(sf_col, tk.Canvas)
    canvas_col[0].configure(bg = bg_base_color) 

    inner_frame_col = sf_col.display_widget(tk.Frame, bg = bg_base_color)
    
    lbl_feat = tk.Label(inner_frame_col, text = "Features: ", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
    lbl_feat.pack(anchor = "w")
    
    checks_col = []
    for col in df_columns:
        var_col = tk.BooleanVar()
        ckb_col = tk.Checkbutton(inner_frame_col, text = col, font = font,
                              fg = fg_base_color, bg = bg_base_color,
                              selectcolor = bg_base_color, variable = var_col)
        ckb_col.config(activebackground = bg_base_color)
        ckb_col.config(activeforeground = "#ffffff")
        ckb_col.pack(anchor = "w")
        checks_col.append((col, var_col))
    
    selected_label = tk.StringVar()

    cb_label = ttk.Combobox(Frame_col, values = df_columns, font = font,
                            textvariable = selected_label,
                            state = "readonly", width = 12)
    cb_label.set(df_columns[-1])
    cb_label.place(x = 50, y = 10)
    
    def select_all_col():
        if var_col_all.get() == 1:
            for i in checks_col:
                i[1].set(1)
        else:
            for i in checks_col:
                i[1].set(0)
    
    var_col_all = tk.BooleanVar()
    ckb_all = tk.Checkbutton(Frame_col, text = "Select All", font = font,
                          fg = fg_base_color, bg = bg_base_color,
                          selectcolor = bg_base_color, variable = var_col_all,
                          command = select_all_col)
    ckb_all.config(activebackground = bg_base_color)
    ckb_all.config(activeforeground = "#ffffff")
    ckb_all.place(x = 0, y = 170)
    
    def df_new_create():
        global df_feat, df_label, selected_cols
        selected_cols = [col for col, var_col in checks_col if var_col.get()]
        
        # Verifies if there is at least 2 variables to the model work with
        if len(selected_cols) < 2:
            messagebox.showerror("Error", "You have to select at least 2 variables")
            return
        
        if selected_label.get() in selected_cols:
            selected_cols.remove(selected_label.get())

        df_feat = df[selected_cols]
        df_label = df[selected_label.get()]
        
        df_new = pd.concat([df_feat, df_label], axis=1)
    
        root_col.destroy()
        Update_pickle(Temp_path,{"df_new": df_new})
        
    btn_col = tk.Button(Frame_col, text = "Confirm variables", font = font,
                        bg = entr_btn_color, fg = fg_base_color,
                        command = df_new_create)
    btn_col.place(x = 90, y = 170)
    
    root.wait_window(root_col)

# Tools Functions==============================================================
def Predict_y():
    with open(Temp_path, "rb") as file:
        variables = pickle.load(file)
        model = variables["model"]
        parms_X = variables["parms_X"]
        parms_Y = variables["parms_Y"]
    
    var_norm_label = str(selected_norm_label.get())
    var_norm_feat = str(selected_norm_feat.get())
    
    x_value = np.array([float(e.get()) for e in x_value_entry]).reshape((1,-1))
    x_value_N = Normalization(x_value, var_norm_feat, parms = parms_X,
                             reverse = False)
    
    Y_mean_N, Y_var_N = model.predict_y(x_value_N[0], full_cov = False)
    
    Y_mean = Normalization(np.array([Y_mean_N]).reshape(-1,1),
                           var_norm_label, parms = parms_Y, reverse = True)
    Y_var = Normalization(np.array([Y_var_N]).reshape(-1,1), var_norm_label,
                          parms = parms_Y, reverse = True,
                          var = {"bol": True, "Y_N": Y_mean_N})
    
    lbl3_Pred.config(text = round(float(Y_mean[0]), 3))
    
    conf_level = float(CI_percent_entry.get()) / 100
    z_score = norm.ppf(1 - (1 - conf_level) / 2)
    
    Y_Upper_value = Y_mean[0] + z_score * np.sqrt(Y_var[0])
    Y_Lower_value = Y_mean[0] - z_score * np.sqrt(Y_var[0])

    lbl5_Pred.config(text = "[" + str(round(float(Y_Lower_value), 3)) + " , " +
                str(round(float(Y_Upper_value), 3)) + "]")

def Next_measure_ALBO():
    global BO_zone
    with open(Temp_path, "rb") as file:
        variables = pickle.load(file)
        var_norm_label = variables["var_norm_label"]
        var_norm_feat = variables["var_norm_feat"]
        model = variables["model"]
        parms_X = variables["parms_X"]
        parms_Y = variables["parms_Y"]
        ncol_data = variables["ncol_data"]
    
    # Selected Aquisition Functions (AF) type
    var_AF = str(selected_AF.get())
    
    # AF searching zone with or without available data
    if var_avail_ALBO.get() == 0:
        
        # Features min and max values (depending if standard plot is active or not)
        if var_sel_ALBO.get() == 1:
            with open(Temp_path, "rb") as file:
                    variables = pickle.load(file)
                    X = variables["X"]
                    
            x_min = np.array([np.min(X[:,e]) for e in range(ncol_data-1)]).reshape((-1,1))
            x_max = np.array([np.max(X[:,e]) for e in range(ncol_data-1)]).reshape((-1,1))
            
        else:
            x_min = np.array([float(e.get()) for e in x_min_entry_ALBO]).reshape((-1,1))
            x_max = np.array([float(e.get()) for e in x_max_entry_ALBO]).reshape((-1,1))
        
        N_Points = round(1000 ** (1 / (ncol_data - 1)))
        BO_zone = np.zeros((N_Points, ncol_data - 1))
        
        for n in range(ncol_data - 1):
            varRange = np.linspace(float(x_min[n]), float(x_max[n]), N_Points)
            BO_zone[:,n] = varRange.copy()
            
        BO_zone = np.array(np.meshgrid(*[BO_zone[:, i] for i in range(ncol_data - 1)]))\
            .T.reshape(-1, ncol_data - 1)
        
        BO_zone_N = Normalization(BO_zone, var_norm_feat, parms = parms_X,
                                 reverse = False)
    else:
        # Asks for the file with the data for the search zone
        file_path = filedialog.askopenfilename(title = "Open CSV File", filetypes = [("CSV files", "*.csv")])
        # Enables the variable selection window and imports from the pickle
        # file the new dataset
        Select_var(file_path)
        with open(Temp_path, "rb") as file:
                variables = pickle.load(file)
                df_new = variables["df_new"]
        
        # Division of the new dataset into features (X) and label (Y)
        ncol_data = df_new.shape[1]
        X = df_new.iloc[:,:-1].values[:, None].reshape(-1, ncol_data-1)
        BO_zone = X.astype(float)
        # Normalizes the features available for search
        BO_zone_N = Normalization(BO_zone, var_norm_feat, parms = parms_X,
                                 reverse = False)
        
    Y_mean_N, Y_var_N = model.predict_y(BO_zone_N[0], full_cov = False)
    
    Y_mean = Normalization(np.array([Y_mean_N]).reshape(-1,1),
                           var_norm_label, parms = parms_Y, reverse = True)
    
    Y_var = Normalization(np.array([Y_var_N]).reshape(-1,1), var_norm_label,
                          parms = parms_Y, reverse = True,
                          var = {"bol": True, "Y_N": Y_mean_N})
    
    # Types of AF and their calculation
    if var_AF == "PI":
        AF = norm.cdf((Y_mean[0] - max(Y_mean[0])) / Y_var[0], 0 , 1)
    elif var_AF == "EI":
        AF = (Y_mean[0] - max(Y_mean[0]))\
            * norm.cdf((Y_mean[0] - max(Y_mean[0])) / Y_var[0], 0 , 1) + Y_var[0]\
                * norm.pdf((Y_mean[0] - max(Y_mean[0])) / Y_var[0], 0 , 1)
    elif var_AF == "UCB":
        lamda = 1
        AF = Y_mean[0] + lamda * Y_var[0]
    elif var_AF == "Std":
        AF = np.sqrt(Y_var[0])
    elif var_AF == "Std/Mean":
        AF = np.sqrt(Y_var[0])/Y_mean[0]
    
    # Searching for the max of the AF (every points)
    max_AF = float(max(AF))
    index_AF = np.where(AF == max_AF)[0]
    next_point = BO_zone[index_AF]
    
    # Display results on the ALBO interface
    for i in range(ncol_data - 1):
        lbl_ALBO_Results[i].configure(text = str(round(float(next_point[0,i]), 3)))

# File Functions===============================================================
def Save_as():
    Save_path = filedialog.asksaveasfilename(defaultextension = ".pkl",
        filetypes = [("Pickle files", "*.pkl"), ("All files", "*.*")])  
    if not Save_path:
        return
    
    with open(Temp_path, "rb") as file:
            variables = pickle.load(file)
    
    with open(Save_path, "wb") as file:
            pickle.dump(variables, file)
    
    Update_pickle(Temp_path, {"Save_path": Save_path})

def Save():
    with open(Temp_path, "rb") as file:
            variables = pickle.load(file)
            Save_path = variables["Save_path"]
    
    if "Save_path" in variables:
        del variables["Save_path"]
    
    with open(Save_path, "wb") as file:
            pickle.dump(variables, file)

def Load():
# Loading Process==============================================================
    Save_path = filedialog.askopenfilename(defaultextension = ".pkl",
        filetypes = [("Pickle files", "*.pkl"), ("All files", "*.*")])
    
    with open(Save_path, "rb") as file:
            variables = pickle.load(file)
            
    
    with open(Temp_path, "wb") as file:
             pickle.dump(variables, file)   
    
    Update_pickle(Temp_path, {"Save_path": Save_path})
    
    with open(Temp_path, "rb") as file:
            variables = pickle.load(file)
            ncol_data = variables["ncol_data"]
            X_Train = variables["X_Train"]
            X_Test = variables["X_Test"]
            trainlikelihood = variables["trainlikelihood"]
            Train_Test_Split = variables["Train_Test_Split"]
            Split_Percentage = variables["Split_Percentage"]
            var_norm_label = variables["var_norm_label"]
            var_norm_feat = variables["var_norm_feat"]
            var_kernel = variables["var_kernel"]
            num_params = variables["num_params"]
            Axes_titles = variables["Axes_titles"]
            white_kernel = variables["white_kernel"]
            
# Entries Creation=============================================================
    # Destroys and clears the old "Predict Y Window" entries
    for i in range(0,len(x_value_entry)):
        x_value_entry[i].destroy()
        lbl_Pred_Titles[i].destroy()
    x_value_entry.clear()
    lbl_Pred_Titles.clear()
    
    # Creates new "Predict Y Window" entries
    for i in range(0,ncol_data-1):
        x_value_entry.append(tk.Entry(inner_frame_Pred, width = 12, bg = entr_btn_color,
                              fg = fg_base_color,
                              insertbackground = fg_base_color))
        x_value_entry[i].grid(row = 1, column = i, padx=5, pady=2)
        
        lbl_Pred_Titles.append(tk.Label(inner_frame_Pred, text = "", font = font,
                             bg = bg_base_color, fg = fg_base_color))
        lbl_Pred_Titles[i].grid(row = 0, column = i, padx=5, pady=2)
    
    # Destroys and clears the old "AL and BO Window" entries
    for i in range(0,len(x_min_entry_ALBO)):
        x_min_entry_ALBO[i].destroy()
        x_max_entry_ALBO[i].destroy()
        lbl_ALBO_Titles[i].destroy()
        lbl_ALBO_Results[i].destroy()
    x_min_entry_ALBO.clear()
    x_max_entry_ALBO.clear()
    lbl_ALBO_Titles.clear()
    lbl_ALBO_Results.clear()

    # Creates new "Bayesian Optimization Window" entries
    for i in range(0,ncol_data-1):
        x_min_entry_ALBO.append(tk.Entry(inner_frame_ALBO, width = 10, bg = entr_btn_color,
                              fg = fg_base_color,
                              insertbackground = fg_base_color))
        x_min_entry_ALBO[i].grid(row = i, column = 1, padx=5, pady=2)
        x_max_entry_ALBO.append(tk.Entry(inner_frame_ALBO, width = 10, bg = entr_btn_color,
                              fg = fg_base_color,
                              insertbackground = fg_base_color))
        x_max_entry_ALBO[i].grid(row = i, column = 2, padx=5, pady=2)
        
        lbl_ALBO_Titles.append(tk.Label(inner_frame_ALBO, text = "", font = font,
                             bg = bg_base_color, fg = fg_base_color))
        lbl_ALBO_Titles[i].grid(row = i, column = 0, padx=5, pady=2, sticky="w")

        lbl_ALBO_Results.append(tk.Label(inner_frame_ALBO, text = "", font = font,
                             bg = bg_base_color, fg = fg_base_color))
        lbl_ALBO_Results[i].grid(row = i, column = 3, padx=5, pady=2, sticky="w")
    
    if ncol_data == 2:
        entry_plot_min[0].configure(state = "normal")
        entry_plot_max[0].configure(state = "normal")
        lbl_Plot_Titles[0].configure(text = Axes_titles[0] + " :")
    elif ncol_data == 3:
        for i in range(0,2):
            entry_plot_min[i].configure(state = "normal")
            entry_plot_max[i].configure(state = "normal")
            lbl_Plot_Titles[i].configure(text = Axes_titles[i] + " :")
    elif ncol_data > 3 or ncol_data == 1:
        for i in range(0,2):
            entry_plot_min[i].configure(state = "disabled")
            entry_plot_max[i].configure(state = "disabled")
            lbl_Plot_Titles[i].configure(text = "Variable" + str(i+1) + ":")
        
# Set configs and update labels================================================
    if trainlikelihood:
        checkvar1.set(1)
    elif not trainlikelihood:
        checkvar1.set(0)
    
    if white_kernel:
        checkvar7.set(1)
    elif not white_kernel:
        checkvar7.set(0)
    
    if Train_Test_Split:
        checkvar2.set(1)
        Test_Percentage.configure(state = "normal")
        Test_Percentage.delete(0, tk.END)
        Test_Percentage.insert(0, str(Split_Percentage))
    elif not Train_Test_Split:
        checkvar2.set(0)
        Test_Percentage.delete(0, tk.END)

    cb1.set(var_kernel)
    cb2.set(var_norm_label)
    cb3.set(var_norm_feat)
    
    # Updates "Frame_info" with the new model information
    lbl15.configure(text = ncol_data)
    lbl16.configure(text = np.size(X_Train,0))
    lbl17.configure(text = np.size(X_Test,0))
    lbl18.configure(text = num_params)
    
    # Updates Predict Y Window labels
    for i in range(ncol_data - 1):
        lbl_Pred_Titles[i].configure(text = Axes_titles[i])
    
    lbl2_Pred.configure(text = "Pred. " + Axes_titles[-1] + " :")
    
    # Updates AL and BO Window labels
    for i in range(ncol_data - 1):
        lbl_ALBO_Titles[i].configure(text = Axes_titles[i] + " :")
    
# Disabling Widgets============================================================
    bt1.configure(state = "disabled")    

    # Disables widgets no longer necessary
    for child in Frame_kernel_norm.winfo_children():
        child.configure(state = 'disabled')
    
    for child in Frame_split.winfo_children():
        child.configure(state = 'disabled')
        
    bt2.configure(state = "disabled")
    
    for i in range(rows):
        for j in range(cols):
            entry = entries[i][j]
            entry.config(state = "disabled")

# =============================================================================
# MAIN SCRIPT
# =============================================================================

# Main Window==================================================================
root = tk.Tk()

# Ativa a perceção de DPI no Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

root.title("GP Training App")
root.geometry("1200x800")
root.minsize(1000, 700)
root.resizable(True, True)
root.configure(borderwidth = 10, bg = background_color)

# Frames=======================================================================
Frame_points = tk.Frame(root, width = 200, height = 170, bg = bg_base_color)
Frame_points.place(x = 10, y = 10)

Frame_kernel_norm = tk.Frame(root, width = 200, height = 160, bg = bg_base_color)
Frame_kernel_norm.place(x = 10, y = 250)

Frame_import = tk.Frame(root, width = 200, height = 50, bg = bg_base_color)
Frame_import.place(x = 10, y = 190)

Frame_split = tk.Frame(root, width = 200, height = 80, bg = bg_base_color)
Frame_split.place(x = 10, y = 420)

Frame_parity = tk.Frame(root, width = 470, height = 350, bg = bg_base_color)
Frame_parity.place(x = 220, y = 10)

Frame_Train = tk.Frame(root, width = 125, height = 130, bg = bg_base_color)
Frame_Train.place(x = 220, y = 370)

Frame_info = tk.Frame(root, width = 195, height = 130, bg = bg_base_color)
Frame_info.place(x = 355, y = 370)

Frame_icon = tk.Frame(root, width = 130, height = 130, bg = bg_base_color)
Frame_icon.place(x = 560, y = 370)

# Labels=======================================================================
lbl1 = tk.Label(Frame_points, text = "Training Points", font = font, 
                bg = bg_base_color, fg = fg_base_color)
lbl1.place(x = 55, y = 10)

lbl2 = tk.Label(Frame_points, text = "X", font = font,
                bg = bg_base_color, fg = fg_base_color)
lbl2.place(x = 55, y = 30)

lbl3 = tk.Label(Frame_points, text = "Y", font = font, bg = bg_base_color,
                fg = fg_base_color)
lbl3.place(x = 130, y = 30)

lbl4 = tk.Label(Frame_kernel_norm, text = "Kernel :", font = font, 
                bg = bg_base_color, fg = fg_base_color)
lbl4.place(x = 10, y = 10)

lbl5 = tk.Label(Frame_kernel_norm, text = "Normalization :", font = font,
                 bg = bg_base_color, fg = fg_base_color)
lbl5.place(x = 10, y = 70)

lbl6 = tk.Label(Frame_kernel_norm, text = "-Label :", font = font,
                 bg = bg_base_color, fg = fg_base_color)
lbl6.place(x = 15, y = 100)

lbl7 = tk.Label(Frame_kernel_norm, text = "-Features :", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl7.place(x = 15, y = 130)

lbl8 = tk.Label(Frame_split, text = "Test percentage :", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl8.place(x = 10, y = 40)

lbl9 = tk.Label(Frame_split, text = "%", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl9.place(x = 175, y = 40)

lbl10 = tk.Label(Frame_parity, text = "Options :", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl10.place(x = 10, y = 310)

lbl11 = tk.Label(Frame_info, text = "Nº of Dimensions :", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl11.place(x = 10, y = 40)

lbl12 = tk.Label(Frame_info, text = "Nº of Training points :", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl12.place(x = 10, y = 60)

lbl13 = tk.Label(Frame_info, text = "Nº of Testing points :", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl13.place(x = 10, y = 80)

lbl14 = tk.Label(Frame_info, text = "Nº of Hyperparameters :", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl14.place(x = 10, y = 100)

lbl15 = tk.Label(Frame_info, text = " --- ", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl15.place(x = 145, y = 40)

lbl16 = tk.Label(Frame_info, text = " --- ", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl16.place(x = 145, y = 60)

lbl17 = tk.Label(Frame_info, text = " --- ", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl17.place(x = 145, y = 80)

lbl18 = tk.Label(Frame_info, text = " --- ", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl18.place(x = 145, y = 100)

lbl19 = tk.Label(Frame_info, text = "Model details", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl19.place(x = 55, y = 10)

img_path = os.path.join(os.path.dirname(__file__), "MACACO_DO_GP.png")
img = Image.open(img_path)
photo = tk.PhotoImage(file = img_path, master = Frame_icon)
panel = tk.Label(Frame_icon, image = photo)
panel.image = photo
panel.place(relwidth = 1, relheight = 1)

# Point Entries================================================================
inpt1 = []
inpt2 = []
rows = 5
cols = 2

for i in range(0,rows):
    inpt1.append(tk.Entry(Frame_points, width = 10, bg = entr_btn_color,
                          fg = fg_base_color,
                          insertbackground = fg_base_color))
    inpt1[i].place(x = 30, y = 50 + i * 22)
    inpt2.append(tk.Entry(Frame_points, width = 10, bg = entr_btn_color,
                          fg = fg_base_color,
                          insertbackground = fg_base_color))
    inpt2[i].place(x = 105, y = 50 + i * 22)

entries = []

for i in range(len(inpt1)):
    line = [inpt1[i], inpt2[i]]
    entries.append(line)

# Bind for all manual training point entries (so arrow keys can be used)
for i in range(rows):
    for j in range(cols):
        entry = entries[i][j]
        entry.bind("<Up>", lambda e, r=i, c=j: move_focus(entries, r, c, -1, 0))
        entry.bind("<Down>", lambda e, r=i, c=j: move_focus(entries, r, c, 1, 0))
        entry.bind("<Left>", lambda e, r=i, c=j: move_focus(entries, r, c, 0, -1))
        entry.bind("<Right>", lambda e, r=i, c=j: move_focus(entries, r, c, 0, 1))

# Entry for the Test split percentage==========================================
Test_Percentage = tk.Entry(Frame_split, width = 8, bg = entr_btn_color, font = font,
                           fg = fg_base_color, insertbackground = fg_base_color,
                           state = "disabled")
Test_Percentage.place(x = 115, y = 40)

# Buttons======================================================================
bt1 = tk.Button(Frame_Train, text = "Train Model", command = Train, width = 13,
                font = font, bg = entr_btn_color, fg = fg_base_color)
bt1.place(x = 10, y = 10)

bt2 = tk.Button(Frame_import, text = "Import CSV File",
                command = Open_CSV_File, width = 19, font = font,
                bg = entr_btn_color, fg = fg_base_color)
bt2.place(x = 30, y = 12)

bt3 = tk.Button(Frame_Train, text = "Global Reset", command = Global_Reset,
                width = 13, font = font, bg = entr_btn_color,
                fg = fg_base_color)
bt3.place(x = 10, y = 90)

bt4 = tk.Button(Frame_parity, text = "Plot Parity", command = Plot_parity, width = 10,
                font = font, bg = entr_btn_color, fg = fg_base_color)
bt4.place(x = 380, y = 310)

bt5 = tk.Button(Frame_Train, text = "Config. Reset", command = Config_Reset,
                width = 13, font = font, bg = entr_btn_color,
                fg = fg_base_color)
bt5.place(x = 10, y = 50)

# Comboboxes for the types of kernel and normalization=========================
types_kernel = ["RBF", "Matern 12", "Matern 32", "Matern 52", "Periodic", "RQ"]
selected_kernel = tk.StringVar()

cb1 = ttk.Combobox(Frame_kernel_norm, values = types_kernel, font = font,
                   textvariable = selected_kernel,
                   state = "readonly", width = 12)
cb1.set("RBF")
cb1.place(x = 80, y = 10)

types_normalization = ["None", "Standardization", "LogStand", "Log + bStand",
                       "MinMax"]
selected_norm_label = tk.StringVar()

cb2 = ttk.Combobox(Frame_kernel_norm, values = types_normalization,
                   font = font, textvariable = selected_norm_label,
                   state = "readonly", width = 12)
cb2.set("None")
cb2.place(x = 80, y = 100)

selected_norm_feat = tk.StringVar()

cb3 = ttk.Combobox(Frame_kernel_norm, values = types_normalization,
                   font = font, textvariable = selected_norm_feat,
                   state = "readonly", width = 12)
cb3.set("None")
cb3.place(x = 80, y = 130)

# CheckButton==================================================================
checkvar1 = tk.IntVar()
ckb1 = tk.Checkbutton(Frame_kernel_norm, text = 'Likelihood',
                      font = font, fg = fg_base_color, bg = bg_base_color,
                      selectcolor = bg_base_color, variable = checkvar1,
                      onvalue = 1, offvalue = 0)
ckb1.config(activebackground = bg_base_color)
ckb1.config(activeforeground = "#ffffff")
ckb1.place(x = 10, y = 40)

checkvar2 = tk.IntVar()
ckb2 = tk.Checkbutton(Frame_split, text = 'Split', font = font,
                      fg = fg_base_color, bg = bg_base_color,
                      selectcolor = bg_base_color, variable = checkvar2,
                      onvalue = 1, offvalue = 0,
                      command = Do_Split)
ckb2.config(activebackground = bg_base_color)
ckb2.config(activeforeground = "#ffffff")
ckb2.place(x = 10, y = 10)

checkvar3 = tk.IntVar()
ckb3 = tk.Checkbutton(Frame_parity, text = 'MAE', font = font,
                      fg = fg_base_color, bg = bg_base_color,
                      selectcolor = bg_base_color, variable = checkvar3,
                      onvalue = 1, offvalue = 0)
ckb3.config(activebackground = bg_base_color)
ckb3.config(activeforeground = "#ffffff")
ckb3.place(x = 70, y = 310)

checkvar4 = tk.IntVar()
ckb4 = tk.Checkbutton(Frame_parity, text = 'MAPE', font = font,
                      fg = fg_base_color, bg = bg_base_color,
                      selectcolor = bg_base_color, variable = checkvar4,
                      onvalue = 1, offvalue = 0)
ckb4.config(activebackground = bg_base_color)
ckb4.config(activeforeground = "#ffffff")
ckb4.place(x = 125, y = 310)

checkvar5 = tk.IntVar()
ckb5 = tk.Checkbutton(Frame_parity, text = 'R²', font = font,
                      fg = fg_base_color, bg = bg_base_color,
                      selectcolor = bg_base_color, variable = checkvar5,
                      onvalue = 1, offvalue = 0)
ckb5.config(activebackground = bg_base_color)
ckb5.config(activeforeground = "#ffffff")
ckb5.place(x = 185, y = 310)

checkvar6 = tk.IntVar()
ckb6 = tk.Checkbutton(Frame_parity, text = 'RMSE', font = font,
                      fg = fg_base_color, bg = bg_base_color,
                      selectcolor = bg_base_color, variable = checkvar6,
                      onvalue = 1, offvalue = 0)
ckb6.config(activebackground = bg_base_color)
ckb6.config(activeforeground = "#ffffff")
ckb6.place(x = 230, y = 310)

checkvar7 = tk.IntVar()
ckb7 = tk.Checkbutton(Frame_kernel_norm, text = 'White Kernel',
                      font = font, fg = fg_base_color, bg = bg_base_color,
                      selectcolor = bg_base_color, variable = checkvar7,
                      onvalue = 1, offvalue = 0)
ckb7.config(activebackground = bg_base_color)
ckb7.config(activeforeground = "#ffffff")
ckb7.place(x = 95, y = 40)

checkvar8 = tk.IntVar()
ckb8 = tk.Checkbutton(Frame_parity, text = 'Error Bars',
                      font = font, fg = fg_base_color, bg = bg_base_color,
                      selectcolor = bg_base_color, variable = checkvar8,
                      onvalue = 1, offvalue = 0)
ckb8.config(activebackground = bg_base_color)
ckb8.config(activeforeground = "#ffffff")
ckb8.place(x = 290, y = 310)

# Inicialize empty plot========================================================
Plot_parity()

# Plot graph Window============================================================
root_plot = tk.Toplevel(root)
root_plot.title("GP Plot")
root_plot.geometry("780x405")
root_plot.resizable(False, False)
root_plot.configure(borderwidth = 10, bg = background_color)
root_plot.protocol("WM_DELETE_WINDOW", lambda: Hide_window(root_plot))
Hide_window(root_plot)
#root_plot.iconbitmap("Adobe Express - file.ico")

Frame_plot = tk.Frame(root_plot, width = 470, height = 365, bg = bg_base_color)
Frame_plot.place(x = 10, y = 10)

Plot_graph()

Frame_custom_plot = tk.Frame(root_plot, width = 260, height = 365, bg = bg_base_color)
Frame_custom_plot.place(x = 490, y = 10)

Frame_sel_plot = tk.Frame(Frame_custom_plot, bg = bg_base_color)
Frame_sel_plot.place(x = 10, y = 10)

lbl1_Plot = tk.Label(Frame_custom_plot, text = "Nº of Plot Points :", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl1_Plot.place(x = 10, y = 100)

lbl2_Plot = tk.Label(Frame_custom_plot, text = "3D Graph Style :", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl2_Plot.place(x = 10, y = 130)

lbl4_Plot = tk.Label(Frame_custom_plot, text = "2D Model color :", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl4_Plot.place(x = 10, y = 160)

lbl5_Plot = tk.Label(Frame_custom_plot, text = "2D IC color :", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl5_Plot.place(x = 10, y = 190)

var_sel_plot = tk.BooleanVar()
ckb_std_plot = tk.Checkbutton(Frame_custom_plot, text = "Standard Plot", font = font,
                              fg = fg_base_color, bg = bg_base_color,
                              selectcolor = bg_base_color, variable = var_sel_plot)
ckb_std_plot.config(activebackground = bg_base_color)
ckb_std_plot.config(activeforeground = "#ffffff")
ckb_std_plot.place(x = 10, y = 70)
var_sel_plot.set(1)

btn_plot = tk.Button(Frame_custom_plot, text = "Plot Graph", command = Plot_graph,
                width = 13, font = font, bg = entr_btn_color,
                fg = fg_base_color)
btn_plot.place(x = 10, y = 330)

N_points_plot = ["100", "500", "1000", "2000", "5000", "10000"]
selected_N_points = tk.StringVar()
cb1_plot = ttk.Combobox(Frame_custom_plot, values = N_points_plot, font = font,
                   textvariable = selected_N_points,
                   state = "readonly", width = 12)
cb1_plot.set("1000")
cb1_plot.place(x = 125, y = 100)

Plot_cmap = list(colormaps)
selected_cmap = tk.StringVar()
cb2_plot = ttk.Combobox(Frame_custom_plot, values = Plot_cmap, font = font,
                   textvariable = selected_cmap,
                   state = "readonly", width = 12)
cb2_plot.set("viridis")
cb2_plot.place(x = 125, y = 130)

matplotlib_colors = list(mcolors.CSS4_COLORS.keys())
selected_model_color = tk.StringVar()
cb3_plot = ttk.Combobox(Frame_custom_plot, values = matplotlib_colors, font = font,
                   textvariable = selected_model_color,
                   state = "readonly", width = 12)
cb3_plot.set("black")
cb3_plot.place(x = 125, y = 160)

selected_IC_color = tk.StringVar()
cb3_plot = ttk.Combobox(Frame_custom_plot, values = matplotlib_colors, font = font,
                   textvariable = selected_IC_color,
                   state = "readonly", width = 12)
cb3_plot.set("blue")
cb3_plot.place(x = 125, y = 190)

lbl_Plot_Titles = []
lbl_Plot_Titles.append(tk.Label(Frame_sel_plot, text = "Variable 1 :", font = font,
                     bg = bg_base_color, fg = fg_base_color))
lbl_Plot_Titles[0].grid(row = 0, column = 0, padx=5, pady=2)

lbl_Plot_Titles.append(tk.Label(Frame_sel_plot, text = "Variable 2 :", font = font,
                     bg = bg_base_color, fg = fg_base_color))
lbl_Plot_Titles[1].grid(row = 1, column = 0, padx=5, pady=2)

entry_plot_max = []
entry_plot_max.append(tk.Entry(Frame_sel_plot, width = 8, bg = entr_btn_color,
                              font = font, fg = fg_base_color,
                              insertbackground = fg_base_color, state = "disabled"))
entry_plot_max[0].grid(row = 0, column = 1, padx=5, pady=2)

entry_plot_max.append(tk.Entry(Frame_sel_plot, width = 8, bg = entr_btn_color,
                              font = font, fg = fg_base_color,
                              insertbackground = fg_base_color, state = "disabled"))
entry_plot_max[1].grid(row = 1, column = 1, padx=5, pady=2)

entry_plot_min = []
entry_plot_min.append(tk.Entry(Frame_sel_plot, width = 8, bg = entr_btn_color,
                              font = font, fg = fg_base_color,
                              insertbackground = fg_base_color, state = "disabled"))
entry_plot_min[0].grid(row = 0, column = 2, padx=5, pady=2)

entry_plot_min.append(tk.Entry(Frame_sel_plot, width = 8, bg = entr_btn_color,
                              font = font, fg = fg_base_color,
                              insertbackground = fg_base_color, state = "disabled"))
entry_plot_min[1].grid(row = 1, column = 2, padx=5, pady=2)
# Predict Y Window=============================================================
root_Pred = tk.Toplevel(root)
root_Pred.title("Predict Y")
root_Pred.geometry("375x210")
root_Pred.resizable(False, False)
root_Pred.configure(borderwidth = 10, bg = background_color)
root_Pred.protocol("WM_DELETE_WINDOW", lambda: Hide_window(root_Pred))
Hide_window(root_Pred)
#root_Pred.iconbitmap("Adobe Express - file.ico")

Frame_predict = tk.Frame(root_Pred, width = 335, height = 170, bg = bg_base_color)
Frame_predict.place(x = 10, y = 10)

Frame_Scroll_Pred = tk.Frame(root_Pred, width = 265, height = 75, bg = bg_base_color)
Frame_Scroll_Pred.place(x = 75, y = 10)

sf_Pred = ScrolledFrame(Frame_Scroll_Pred, width=265, height=50, bg = bg_base_color,
                        scrollbars="horizontal", use_ttk=False)
sf_Pred.pack(fill="both", expand=True)

canvas_Pred = find_widgets_by_type(sf_Pred, tk.Canvas)
canvas_Pred[0].configure(bg = bg_base_color) 

inner_frame_Pred = sf_Pred.display_widget(tk.Frame, bg = bg_base_color)

lbl1_Pred = tk.Label(Frame_predict, text = "X value :", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl1_Pred.place(x = 10, y = 10)

lbl2_Pred = tk.Label(Frame_predict, text = "Pred. Y :", font = font,
                     bg = bg_base_color, fg = fg_base_color)
lbl2_Pred.place(x = 10, y = 75)

lbl3_Pred = tk.Label(Frame_predict, text = "", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl3_Pred.place(x = 130, y = 75)

lbl4_Pred = tk.Label(Frame_predict, text = "C.I. :", font = font,
                     bg = bg_base_color, fg = fg_base_color)
lbl4_Pred.place(x = 10, y = 105)

lbl5_Pred = tk.Label(Frame_predict, text = "", font = font,
                     bg = bg_base_color, fg = fg_base_color)
lbl5_Pred.place(x = 130, y = 105)

lbl6_Pred = tk.Label(Frame_predict, text = "Confidence :", font = font,
                     bg = bg_base_color, fg = fg_base_color)
lbl6_Pred.place(x = 10, y = 135)

lbl7_Pred = tk.Label(Frame_predict, text = "%", font = font,
                    bg = bg_base_color, fg = fg_base_color)
lbl7_Pred.place(x = 135, y = 135)

lbl_Pred_Titles = []
lbl_Pred_Titles.append(tk.Label(inner_frame_Pred, text = "Variable 1", font = font,
                     bg = bg_base_color, fg = fg_base_color))
lbl_Pred_Titles[0].grid(row = 0, column = 1, padx=5, pady=2)

# Button to predict Y value
bt1_Pred = tk.Button(Frame_predict, text = "Predict Y", command = Predict_y, width = 7,
                     font = font_btn, bg = entr_btn_color, fg = fg_base_color)
bt1_Pred.place(x = 10, y = 40)

# Entry for the x value selected by the user
x_value_entry = []
x_value_entry.append(tk.Entry(inner_frame_Pred, width = 10, bg = entr_btn_color,
                              font = font, fg = fg_base_color,
                              insertbackground = fg_base_color, state = "disabled"))
x_value_entry[0].grid(row = 1, column = 1, padx=5, pady=2)

CI_percent_entry = tk.Entry(Frame_predict, width = 5, bg = entr_btn_color,
                            font = font, fg = fg_base_color,
                            insertbackground = fg_base_color)
CI_percent_entry.place(x = 95, y = 135)
CI_percent_entry.insert(0, str(95))

# AL and BO Window=================================================
root_ALBO = tk.Toplevel(root)
root_ALBO.title("AL and BO")
#root_ALBO.geometry("940x405")
root_ALBO.resizable(False, False)
root_ALBO.configure(borderwidth = 10, bg = background_color)
root_ALBO.protocol("WM_DELETE_WINDOW", lambda: Hide_window(root_ALBO))
Hide_window(root_ALBO)
#root_ALBO.iconbitmap("Adobe Express - file.ico")

Frame_AF_Plot = tk.Frame(root_ALBO, width = 470, height = 365, bg = bg_base_color)
Frame_AF_Plot.grid(row=0, column=0, rowspan=2, padx=5)

Plot_AF()

Frame_custom_ALBO = tk.Frame(root_ALBO, width = 420, height = 190, bg = bg_base_color)
Frame_custom_ALBO.grid(row=1, column=1, padx=5, pady=5)
Frame_custom_ALBO.grid_propagate(False)

Frame_ALBO = tk.Frame(root_ALBO, width = 420, height = 165, bg = bg_base_color)
Frame_ALBO.grid(row=0, column=1, padx=5, pady=5)

Frame_Scroll_ALBO = tk.Frame(Frame_ALBO, width = 400, height = 90, bg = bg_base_color)
Frame_Scroll_ALBO.place(x = 10, y = 40)

sf_ALBO = ScrolledFrame(Frame_Scroll_ALBO, width = 380, height = 90,
                        scrollbars = "vertical", use_ttk = False)
sf_ALBO.pack(fill = "both", expand = True)

canvas_ALBO = find_widgets_by_type(sf_ALBO, tk.Canvas)
canvas_ALBO[0].configure(bg = bg_base_color)

inner_frame_ALBO = sf_ALBO.display_widget(tk.Frame, bg = bg_base_color)

inner_frame_ALBO.grid_columnconfigure(0, minsize=95)

lbl1_ALBO = tk.Label(Frame_ALBO, text = "X Limits", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl1_ALBO.place(x = 10, y = 10)

lbl2_ALBO = tk.Label(Frame_ALBO, text = "Point to measure :", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl2_ALBO.place(x = 260, y = 10)

lbl3_ALBO = tk.Label(Frame_ALBO, text = "Min.               Max.", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl3_ALBO.place(x = 125, y = 10)

lbl4_ALBO = tk.Label(Frame_custom_ALBO, text = "Aquisition function :", font = font, 
                 bg = bg_base_color, fg = fg_base_color)
lbl4_ALBO.grid(row=0, column=1, columnspan=2, pady=5)

lbl5_ALBO = tk.Label(Frame_custom_ALBO, text = "Nº of Plot Points :", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl5_ALBO.grid(row=1, column=1, columnspan=2, pady=5)

lbl6_ALBO = tk.Label(Frame_custom_ALBO, text = "3D Graph Style :", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl6_ALBO.grid(row=2, column=1, columnspan=2, pady=5)

lbl7_ALBO = tk.Label(Frame_custom_ALBO, text = "2D Model color :", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl7_ALBO.grid(row=3, column=1, columnspan=2, pady=5)

lbl8_ALBO = tk.Label(Frame_custom_ALBO, text = "2D AF color :", font = font,
                     bg = bg_base_color,fg = fg_base_color)
lbl8_ALBO.grid(row=4, column=1, columnspan=2, pady=5)

lbl_ALBO_Titles = []
lbl_ALBO_Titles.append(tk.Label(inner_frame_ALBO, text = "Variable 1 :", font = font,
                     bg = bg_base_color, fg = fg_base_color))
lbl_ALBO_Titles[0].grid(row = 0, column = 0, padx=5, pady=2, sticky="w")

lbl_ALBO_Results = []
lbl_ALBO_Results.append(tk.Label(inner_frame_ALBO, text = "", font = font,
                     bg = bg_base_color, fg = fg_base_color))
lbl_ALBO_Results[0].grid(row = 0, column = 3, padx=5, pady=2, sticky="w")

bt1_ALBO = tk.Button(Frame_ALBO, text = "Search point", command = Next_measure_ALBO,
                     width = 12, font = font, bg = entr_btn_color,
                     fg = fg_base_color)
bt1_ALBO.place(x = 315, y = 140)

bt2_ALBO = tk.Button(Frame_custom_ALBO, text = "Plot AF", command = Plot_AF,
                     width = 12, font = font, bg = entr_btn_color,
                     fg = fg_base_color)
bt2_ALBO.grid(row=5, column=1, columnspan=2, pady=5)

var_sel_ALBO = tk.BooleanVar()
ckb_std_ALBO = tk.Checkbutton(Frame_ALBO, text = "Standard Plot", font = font,
                              fg = fg_base_color, bg = bg_base_color,
                              selectcolor = bg_base_color, variable = var_sel_ALBO)
ckb_std_ALBO.config(activebackground = bg_base_color)
ckb_std_ALBO.config(activeforeground = "#ffffff")
ckb_std_ALBO.place(x = 10, y = 140)
var_sel_ALBO.set(1)

var_avail_ALBO = tk.BooleanVar()
ckb_avail_ALBO = tk.Checkbutton(Frame_ALBO, text = "Import Available Data", font = font,
                              fg = fg_base_color, bg = bg_base_color,
                              selectcolor = bg_base_color, variable = var_avail_ALBO)
ckb_avail_ALBO.config(activebackground = bg_base_color)
ckb_avail_ALBO.config(activeforeground = "#ffffff")
ckb_avail_ALBO.place(x = 150, y = 140)
var_avail_ALBO.set(1)

types_AF = ["Std", "Std/Mean", "PI", "EI", "UCB"]
selected_AF = tk.StringVar()
cb1_ALBO = ttk.Combobox(Frame_custom_ALBO, values = types_AF,
                   font = font, textvariable = selected_AF,
                   state = "readonly", width = 12)
cb1_ALBO.set("Std")
cb1_ALBO.grid(row=0, column=4, columnspan=2, pady=5)

selected_ALBO_cmap = tk.StringVar()
cb2_ALBO = ttk.Combobox(Frame_custom_ALBO, values = Plot_cmap, font = font,
                   textvariable = selected_ALBO_cmap,
                   state = "readonly", width = 12)
cb2_ALBO.set("viridis")
cb2_ALBO.grid(row=2, column=4, columnspan=2, pady=5)

selected_N_points_ALBO = tk.StringVar()
cb3_ALBO = ttk.Combobox(Frame_custom_ALBO, values = N_points_plot, font = font,
                   textvariable = selected_N_points_ALBO,
                   state = "readonly", width = 12)
cb3_ALBO.set("1000")
cb3_ALBO.grid(row=1, column=4, columnspan=2, pady=5)

selected_model_color_ALBO = tk.StringVar()
cb4_ALBO = ttk.Combobox(Frame_custom_ALBO, values = matplotlib_colors, font = font,
                   textvariable = selected_model_color_ALBO,
                   state = "readonly", width = 12)
cb4_ALBO.set("black")
cb4_ALBO.grid(row=3, column=4, columnspan=2, pady=5)

selected_AF_color = tk.StringVar()
cb5_ALBO = ttk.Combobox(Frame_custom_ALBO, values = matplotlib_colors, font = font,
                   textvariable = selected_AF_color,
                   state = "readonly", width = 12)
cb5_ALBO.set("blue")
cb5_ALBO.grid(row=4, column=4, columnspan=2, pady=5)

x_min_entry_ALBO = []
x_min_entry_ALBO.append(tk.Entry(inner_frame_ALBO, width = 8, bg = entr_btn_color,
                              font = font, fg = fg_base_color,
                              insertbackground = fg_base_color, state = "disabled"))
x_min_entry_ALBO[0].grid(row = 0, column = 1, padx=5, pady=2)

x_max_entry_ALBO = []
x_max_entry_ALBO.append(tk.Entry(inner_frame_ALBO, width = 8, bg = entr_btn_color,
                              font = font, fg = fg_base_color,
                              insertbackground = fg_base_color, state = "disabled"))
x_max_entry_ALBO[0].grid(row = 0, column = 2, padx=5, pady=2)

# Menus========================================================================
menubar = tk.Menu(root)

Filemenu = tk.Menu(menubar, tearoff = 0)
Filemenu.add_command(label = "Load", command = Load)
Filemenu.add_command(label = "Save", command = Save)
Filemenu.add_command(label = "Save as", command = Save_as)
Filemenu.add_separator()
Filemenu.add_command(label = "Exit")
menubar.add_cascade(label = "File", menu = Filemenu)

Toolsmenu = tk.Menu(menubar, tearoff = 0)
Toolsmenu.add_command(label = "Plot", command = lambda: Open_window(root_plot))
Toolsmenu.add_command(label = "Predict Y", command = lambda: Open_window(root_Pred))
Toolsmenu.add_command(label = "AL and BO", command = lambda: Open_window(root_ALBO))
menubar.add_cascade(label = "Tools", menu = Toolsmenu)

Helpmenu = tk.Menu(menubar, tearoff = 0)
menubar.add_cascade(label = "Help", menu = Helpmenu)

root.config(menu=menubar)

root.mainloop()

root = tk.Tk()

dpi = root.winfo_fpixels('1i')
root.tk.call('tk', 'scaling', dpi / 72.0)

root.title("GP Training App")
root.geometry("720x530")
root.resizable(True, True)
root.configure(borderwidth = 10, bg = background_color)