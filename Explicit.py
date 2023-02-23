import numpy as np
from numpy import *
import sympy as sp
from process_latex import process_sympy
import Newton
import sys
import os

class UserFunc:

	def LaTeX2Python(func):
		sys.stdout = open(os.devnull, "w") # this ...
		a = str(process_sympy(func))
		sys.stdout = sys.__stdout__ # ... and this are to prevent printing
		# otherwise you get this error:
		# ANTLR runtime and generated code versions disagree: 4.9.1!=4.7.2
		return a

#	def __init__(self, LaTeX=None, Python=None, bounds=((np.NINF, np.inf), (np.NINF, np.inf), (np.NINF, np.inf))):
	def __init__(self, LaTeX=None, Python=None, bounds=((-10, 10), (-10, 10), (-10, 10))):
		self.bounds = bounds
		if LaTeX:
			self.Update(LaTeX, flavor="LaTeX")
		if Python:
			self.Update(Python, flavor="Python")

	# maybe some input cleaning is in order, otherwise this is the perfect target for an injection attack
	def Update(self, func, flavor):
		if flavor == "LaTeX":
			self.LaTeXfunc = func
			rawPython = UserFunc.LaTeX2Python(func)
			print("Python from LaTeX:", rawPython)
			self._func = compile(rawPython, "userFunc", "eval")
		elif flavor == "Python":
			self.LaTeXfunc = ""
			self._func = compile(func, "userFunc", "eval")
		else:
			raise NotImplementedError(f"\"{flavor}\" flavor not supported")
		self._flavor = flavor

	def eval(self, coords):
		X = coords[0]
		Y = coords[1]
		Z = coords[2]
		return eval(self._func).astype(np.float64)

	def ParallelEval(self, coords):
		X = coords[:, 0]
		Y = coords[:, 1]
		Z = coords[:, 2]
		return eval(self._func).astype(np.float64)
