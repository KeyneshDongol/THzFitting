from tkinter import *

root=Tk()
root.geometry("300x150")

def test(*args):
    x = var1.get()
    y = var2.get()
    if x and y:
        button1.config(state="normal")
    else:
        button1.config(state="disabled")

def win2():
    root.destroy()
    root2 = Tk()
    root2.mainloop()
    
var1 = StringVar(root)
var2 = StringVar(root)

var1.trace("w", test)
var2.trace("w", test)

to_fit = [
    "outgoing wave",
    "transmission, reflection"
]
to_find = [
    "permittivity",
    "plasma & frequency",
    "n & k"
]

button1 = Button(root, text="Next", state="disabled", command=win2)

title1 = Label(root, text="To fit:")
title2 = Label(root, text="To find:")

drop1 = OptionMenu(root, var1, *to_fit)
drop2 = OptionMenu(root, var2, *to_find)

title1.pack(anchor= "w")
drop1.pack(anchor="w")
title2.pack(anchor="w")
drop2.pack(anchor="w")
button1.pack()

root.mainloop()
