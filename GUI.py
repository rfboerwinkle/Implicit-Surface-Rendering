import sys
import math
import time
import tkinter as tk
#import tkinter.font as tkf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg') # for LaTeX
import numpy as np
np.seterr(all="ignore")
"""
gotta ignore errors, otherwise you get warnings to stdout (i think, it might be stderr)

/media/robert/Data/school/ScienceFair2023/3D-Fourier-Transform-Rendering-main/Newton.py:196: RuntimeWarning: invalid value encountered in double_scalars
  OriginalSigns = np.absolute(w) / w

/media/robert/Data/school/ScienceFair2023/3D-Fourier-Transform-Rendering-main/Newton.py:208: RuntimeWarning: divide by zero encountered in true_divide
  dw[Terminated == 0] = np.absolute(w[Terminated == 0] / m[Terminated == 0])

/media/robert/Data/school/ScienceFair2023/3D-Fourier-Transform-Rendering-main/Camera.py:16: RuntimeWarning: invalid value encountered in true_divide
  img /= np.max(img)
"""
import time

#import Newton
#from PIL import Image, ImageTk
# https://github.com/augustt198/latex2sympy

import Camera
import Explicit


SETS_WINDOW_SIZE = "400x405"
SETS_ORDER = ("fov", "resolution", "theta", "phi", "x", "y", "z", "threshold")
SETS_TYPES = (float, int, float, float, float, float, float, float)

if len(sys.argv) >= 2:
	choice = sys.argv[1]
else:
	choice = "0"

if choice == "0":
	# default
	FONT_SIZE = 20
	WINDOW_SIZE = "500x500";
	DEFAULT_FLAVOR = "LaTeX"
	DEFAULT_FUNC = "(X-3) \cdot Y \cdot Z";
	CamInfo = {
		"fov": 140 / 180 * math.pi,
		"resolution": 200,
		"theta": 45 * math.pi / 180,
		"phi": 5 * math.pi / 180,
		"x": 0,
		"y": 0,
		"z": 0,
		"threshold": -0.1,
	}

elif choice == "1":
	# eggcrate
	FONT_SIZE = 20
	WINDOW_SIZE = "1100x900"
	DEFAULT_FLAVOR = "Python"
	DEFAULT_FUNC = "np.sin(X) * np.sin(Y) * np.sin(Z)"
	CamInfo = {
		"fov": 140 / 180 * math.pi,
		"resolution": 600,
		"theta": 45 * math.pi / 180,
		"phi": 5 * math.pi / 180,
		"x": math.pi / 2,
		"y": math.pi / 2,
		"z": math.pi / 2,
		"threshold": -0.2,
	}

elif choice == "2":
	# pebbles
	FONT_SIZE = 20
	WINDOW_SIZE = "1100x900"
	DEFAULT_FLAVOR = "LaTeX"
	DEFAULT_FUNC = "((Z+3)^2 + Y^2)\\frac{(X+11)^3}{20} - 3\\sin((X+11)^{2})"
	CamInfo = {
		"theta": 3.68,
		"phi": 0,
		"fov": 140 / 180 * math.pi,
		"resolution": 600,
		"x": 0,
		"y": 0,
		"z": 0,
		"threshold": 1.2,
	}

elif choice == "3":
	# cylinder
	FONT_SIZE = 20
	WINDOW_SIZE = "900x900";
	DEFAULT_FLAVOR = "Python"
	DEFAULT_FUNC = "sqrt(X**2 + (Y+1)**8 + Z**2)"
	#DEFAULT_FUNC = "\\sqrt{(X-0)^2 + (Y+2)^8 + Z^2}"
	CamInfo = {
		"theta": 1.58539,
		"phi": 0.587266,
		"fov": 80 / 180 * math.pi,
		"resolution": 600,
		"x": -3,
		"y": 2,
		"z": 0,
		"threshold": 0.5,
	}

else:
	print("unknown argument:", choice)
	quit()

def setCamAttributes():
	global Cam
	global CamInfo
	Cam = Camera.camera(CamInfo["theta"], CamInfo["phi"], CamInfo["fov"], (CamInfo["resolution"],)*2, CamInfo["x"], CamInfo["y"], CamInfo["z"])

def getCamAttributes():
	global Cam
	global CamInfo
	CamInfo.update({
		"theta": Cam.theta,
		"phi": Cam.phi,
	})

def setCamSettings():
	global CamInfo
	global SettingsBoxes
	for key in SettingsBoxes:
		if SettingsBoxes[key].winfo_exists():
			SettingsBoxes[key].delete(0, tk.END)
			SettingsBoxes[key].insert(0, str(CamInfo[key]))

def getCamSettings():
	global CamInfo
	global SettingsBoxes
	global SETS_ORDER
	global SETS_TYPES
	newCamInfo = {}
	for i,key in enumerate(SETS_ORDER):
		if SettingsBoxes[key].winfo_exists():
			try:
				newCamInfo[key] = SETS_TYPES[i](SettingsBoxes[key].get())
				SettingsBoxes[key].config(fg="black")
			except ValueError as E:
				SettingsBoxes[key].config(fg="red")
				return False
	CamInfo = newCamInfo
	return True

Func = Explicit.UserFunc(Python="X*0")

def setStatusBar(status=None):
	if status is None:
		status = f"Threshold: {CamInfo['threshold']:.3f}, Theta: {Cam.theta:.3f}, Phi: {Cam.phi:.3f}, Position: {(CamInfo['x'], CamInfo['y'], CamInfo['z'])}" # maybe put resolution at some point
	StatusBar.configure(state='normal') # this ...
	StatusBar.delete("1.0", tk.END)
	StatusBar.insert("1.0", status)
	StatusBar.configure(state='disabled') # ... and this make the text read-only

def updateFunc(event=None):
	status = None
	try:
		Func.Update(FuncInput.get(), Flavor.get())
	except Exception as E:
		status = "LaTeX translator says: " + repr(E)

	if Flavor.get() == "LaTeX":
		tmptext = FuncInput.get() # do i really need the "1.0", tk.END?
		tmptext = "$ "+tmptext+" $"
		fig.clear()
		fig.text(0.1, 0.3, tmptext, fontsize=50)
		try:
			LaTeXCanvas.draw()
		except Exception as E:
			tmptext = "syntax error"
			fig.clear()
			fig.text(0.1, 0.3, tmptext, fontsize=50)
			LaTeXCanvas.draw()
			status = "LaTeX renderer says: " + repr(E)

		LaTeXWidget.pack()

	else:
		fig.clear()
		LaTeXCanvas.draw()
		LaTeXWidget.pack_forget()

	setStatusBar(status)

print("""Controls:
Up/Down: increase/decrease phi
Right/Left: increase/decrease theta
Period/Comma: increase/decrease threshold
""")
def displayKeyPress(event):
	global Cam, CamInfo
	if event.keysym == "Up":
		Cam.rotate(0, 0.1)
		if Cam.phi > math.pi/2:
			Cam.rotate(0, (math.pi/2) - Cam.phi)
	elif event.keysym == "Down":
		Cam.rotate(0, -0.1)
		if Cam.phi < -math.pi/2:
			Cam.rotate(0, (-math.pi/2) - Cam.phi)
	elif event.keysym == "Left":
		Cam.rotate(-0.1, 0)
		if Cam.theta < 0:
			Cam.rotate(math.pi*2, 0)
	elif event.keysym == "Right":
		Cam.rotate(0.1, 0)
		if Cam.theta > math.pi*2:
			Cam.rotate(-math.pi*2, 0)
	elif event.keysym == "period":
		CamInfo["threshold"] += 0.1
	elif event.keysym == "comma":
		CamInfo["threshold"] -= 0.1
#	elif event.keysym == "a":
#		Threshold -= 0.1
	else:
		print(event.keysym)
		return # so updateImg doesn't get called
	getCamAttributes()
	setCamSettings()
	render()
	# TODO: for some reason, it will process the events of keys pressed while rendering before displaying it to the screen, leading to a frozen screen while you repeatedly tap a new movement command
	# make it display each frame while computing the next one, por favor

def updateSize(event=None):
	ViewPort.configure(state="normal")
	ViewPort.delete("1.0", tk.END)
	FuncImageID = ViewPort.image_create("1.0", image=FuncImage, pady=ViewPort.winfo_height()/2-FuncImage.height()/2, padx=ViewPort.winfo_width()/2-FuncImage.width()/2)
	ViewPort.configure(state="disabled")

def numpy2tk(array):
	height, width = array.shape[:2]
	ppm_header = f'P6 {width} {height} 255 '.encode()
	data = ppm_header + array.tobytes()
	return tk.PhotoImage(width=width, height=height, data=data, format='PPM')

# the _=None is to make it work as an event handler
def render(_=None):
	global FuncImage, FuncImageID
	status = None
	print("rendering image... ")
	timer = time.time()
	try:
		raw = Cam.RenderExplicit(Func, CamInfo["threshold"])
	except Exception as E:
		status = "Renderer says: " + repr(E)
		raw = np.zeros((1,1,3), dtype=np.uint8)
	FuncImage = numpy2tk(raw)
	timer = time.time() - timer
	print(timer, "s")
#	FuncImageID = ViewPort.create_image(ViewPort.winfo_width()/2, ViewPort.winfo_height()/2, image=FuncImage)
	setStatusBar(status)
	updateSize()

# the _=None is to make it work as an event handler
def warningMenu(_=None):
	global Window
	global FONT
	global StatusBar
	warnWindow = tk.Toplevel(Window)
#	warnWindow.geometry(WARN_WINDOW_SIZE)
	warnWindow.title("diagnostics")
	text = StatusBar.get("1.0", tk.END)
	tk.Label(warnWindow, font=FONT, text=text).pack()

def settingsMenu():
	global SetsWindow
	if SetsWindow.winfo_exists():
		SetsWindow.destroy()
		return
	global Window
	global FONT
	global CamInfo
	global SETS_ORDER
	global SettingsBoxes
	SetsWindow = tk.Toplevel(Window)
	SetsWindow.geometry(SETS_WINDOW_SIZE)
	SetsWindow.title("settings")
	SettingsBoxes = {}
	for key in SETS_ORDER:
		newFrame = tk.Frame(SetsWindow)
		newLabel = tk.Label(newFrame, font=FONT, text=key+":  ")
		newLabel.pack(side=tk.LEFT)
		SettingsBoxes[key] = tk.Entry(newFrame, font=FONT, justify=tk.RIGHT)
		SettingsBoxes[key].pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
		SettingsBoxes[key].bind("<Return>", settingsApply)
		newFrame.pack(fill=tk.X)
	tk.Button(SetsWindow, text="Apply", command=settingsApply).pack()
	tk.Button(SetsWindow, text="Cancel", command=settingsCancel).pack()
	setCamSettings()

# the _=None is to make it work as an event handler
def settingsApply(_=None):
	if getCamSettings():
		setCamAttributes()
		render()
def settingsCancel():
	global SettingsBoxes
	setCamSettings()
	for key in SettingsBoxes:
		if SettingsBoxes[key].winfo_exists():
			SettingsBoxes[key].config(fg="black") # Make this based on the prev. color maybe?

Window = tk.Tk()
Window.geometry(WINDOW_SIZE)
Window.bind("<Configure>", updateSize)
Window.title("Implicit Surface Renderer")
FONT = tk.font.nametofont("TkDefaultFont")
FONT.configure(size=FONT_SIZE)

# instantiate these so the settings button works right
Cam = None
setCamAttributes()
SettingsBoxes = dict()
SetsWindow = tk.Toplevel(Window)
settingsMenu()
SetsWindow.destroy()


DisplayFrame = tk.Frame(Window)

StatusBar = tk.Text(DisplayFrame, height=1, font=FONT)
StatusBar.configure(selectbackground=StatusBar.cget('bg'), inactiveselectbackground=StatusBar.cget('bg'))
StatusBar.pack(side=tk.TOP, fill=tk.X)
StatusBar.bind("<Button-1>", warningMenu)

ViewPort = tk.Text(DisplayFrame, height=1)
ViewPort.configure(selectbackground=ViewPort.cget('bg'), inactiveselectbackground=ViewPort.cget('bg'))
ViewPort.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
ViewPort.bind("<KeyPress>", displayKeyPress)

DisplayFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


LaTeXWidget = tk.Label(Window)
fig = matplotlib.figure.Figure(figsize=(20, 2), dpi=70)
LaTeXCanvas = FigureCanvasTkAgg(fig, master=LaTeXWidget)
LaTeXCanvas.get_tk_widget().pack(side="top", fill="both", expand=True)
LaTeXCanvas._tkcanvas.pack(side="top", fill="both", expand=True)


FuncInputFrame = tk.Frame(Window)

SettingsButton = tk.Button(FuncInputFrame, text="⚙", command=settingsMenu)
GEAR_FONT = FONT.copy()
GEAR_FONT.configure(size=20)
SettingsButton.config(font=GEAR_FONT)
SettingsButton.pack(side=tk.LEFT)

Flavor = tk.StringVar(Window)
Flavor.set(DEFAULT_FLAVOR)
# this gets called whenever the flavor changes
Flavor.trace("w", lambda name, index, mode: updateFunc())
FlavorMenuButton = tk.OptionMenu(FuncInputFrame, Flavor, "Python", "LaTeX")
# dropdown menus don't inherit fonts i think?
FlavorMenu = Window.nametowidget(FlavorMenuButton.menuname)
FlavorMenu.config(font=FONT)
FlavorMenuButton.pack(side=tk.LEFT)

FuncInput = tk.Entry(FuncInputFrame, font=FONT)
FuncInput.insert(0, DEFAULT_FUNC)
FuncInput.pack(fill=tk.X, side=tk.LEFT, expand=True)
FuncInput.bind("<KeyRelease>", updateFunc)
FuncInput.bind("<Return>", render)

FuncInputFrame.pack(side=tk.BOTTOM, fill=tk.X)

updateFunc()
render()

Window.mainloop()
