#import tkinter as tk
from tkinter import *
from tkinter import scrolledtext as st
import sys
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkinter import messagebox as MessageBox
import os

class Aplicacion:
    def __init__(self):
        self.ventana1=Tk()
        self.agregar_menu()
        self.frame()
        
        #self.scrolledtext1=st.ScrolledText(self.ventana1, width=80, height=20)
        #self.scrolledtext1.grid(column=0,row=0, padx=10, pady=10)
        self.ventana1.mainloop()

    def agregar_menu(self):
        menubar1 = Menu(self.ventana1)
        self.ventana1.config(menu=menubar1)
        opciones1 = Menu(menubar1, tearoff=0)
        opciones1.add_command(label="Recuperar archivo", command=self.recuperar)
        opciones1.add_separator()
        opciones1.add_command(label="Salir", command=self.salir)
        menubar1.add_cascade(label="Archivo", menu=opciones1)  

    def salir(self):
        sys.exit()

    def recuperar(self):
        nombrearch=fd.askopenfilename(initialdir = "/",title = "Seleccione archivo",filetypes = (("txt files","*.txt"),("todos los archivos","*.*")))
        if nombrearch!='':
            MessageBox.showinfo("Data Cargada", "Data Cargada") # título, mensaje

            return nombrearch
        else:
            MessageBox.showerror("Error","NO SE HA SELECCIONADO DATA A CARGAR")
            """
            archi1=open(nombrearch, "r", encoding="utf-8")
            contenido=archi1.read()
            archi1.close()
            self.scrolledtext1.delete("1.0", END) 
            self.scrolledtext1.insert("1.0", contenido)"""
            
    def run(self):
        #ruta=os.getcwd()
        #archivo='sna.py'
        os.system('sna.py')
        
        pass
    
    def network_ressult(self):
        os.system('Predict_sna-1.py')
        pass
    
    def resta(self):
        pass
            
    def frame(self):
        
        
        Label(self.ventana1, text="Data SNA").pack()
        #Entry(self.ventana1, justify="center", textvariable=n1).pack()
        Button(text="CARGAR DATA SNA", command=self.recuperar).pack()
        
        
        Label(self.ventana1, text="Data Otros Modelos").pack()
        #Entry(self.ventana1, justify="center", textvariable=n2).pack()
        Button(self.ventana1, text="Cargar Data usuario", command=self.recuperar).pack()
        
        
        
        Label(self.ventana1, text="").pack()  # Separador
        #archivo='sna.py'
        Button(self.ventana1, text="NETWORK ANALISYS", command=self.run).pack(side="left")
        Button(self.ventana1, text="RESULTADO MODELO", command=self.network_ressult).pack(side="left")
        #Button(self.ventana1, text="NETWORK ANALISYS", command=self.network_ressult).pack(side="left")
        pass


aplicacion1=Aplicacion()