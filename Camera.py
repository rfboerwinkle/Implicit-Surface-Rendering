import math
import numpy as np
import cv2
import Newton
import time

def show(img, time = 0):
	cv2.imshow("image", img)
	cv2.waitKey(time)
	if time == 0:
		cv2.destroyAllWindows()

def normalize(img):
	img = img * 1.0
	img -= np.min(img)
	img /= np.max(img)
	img *= 255.999
	return np.uint8(img)

def cartesian2Spherical(x, y, z):
	return np.arctan2(y,x), np.arctan2(np.sqrt(np.power(x, 2) + np.power(y, 2)), z),  np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))

def spherical2Cartesian(theta, phi, rho = 1):
	return rho * np.cos(theta) * np.sin(phi), rho * np.sin(theta) * np.sin(phi), rho * np.cos(phi)

def getVectorMagnitude(Vect):
	return np.sum(np.power(Vect, 2)) ** 0.5

def normalizeVector(Vect):
	t = np.arctan2(Vect[1], Vect[0])
	p = np.arctan2(np.sqrt(np.power(Vect[0], 2) + np.power(Vect[1], 2)), Vect[2])
	return np.array([np.cos(t) * np.sin(p), np.sin(t) * np.sin(p), np.cos(p)], np.float32)

def parallelNormalizeVector(Vect):
	t = np.arctan2(Vect[:,:,1], Vect[:,:,0])
	p = np.arctan2(np.sqrt(np.power(Vect[:,:,0], 2) + np.power(Vect[:,:,1], 2)), Vect[:,:,2])
	Vect[:,:,0] = np.cos(t) * np.sin(p)
	Vect[:,:,1] = np.sin(t) * np.sin(p)
	Vect[:,:,2] = np.cos(p)
	return Vect

class camera:
	def facePoint(self, coords):
		theta, phi, _ = cartesian2Spherical(coords[0] - self.origin[0], coords[1] - self.origin[1], coords[2] - self.origin[2])
		self.faceDir(theta, phi)

	def faceDir(self, theta, phi):
		self.rotate(theta - self.theta, phi - self.phi)

	def rotate(self, theta, phi):
		self.theta += theta
		self.phi += phi
		R1 = np.array([[math.cos(self.theta), 0, math.sin(self.theta)], [0, 1, 0], [-math.sin(self.theta), 0, math.cos(self.theta)]], np.float32)
		R2 = np.array([[1, 0,  0], [0, math.cos(self.phi), -math.sin(self.phi)], [0, math.sin(self.phi), math.cos(self.phi)]], np.float32)
		Matrix = R1@R2
		self.coords[:,:,0] = self.screen[:, :, 0] * Matrix[0, 0] + self.screen[:, :, 1] * Matrix[0, 1] + self.screen[:, :, 2] * Matrix[0, 2]
		self.coords[:,:,1] = self.screen[:, :, 0] * Matrix[1, 0] + self.screen[:, :, 1] * Matrix[1, 1] + self.screen[:, :, 2] * Matrix[1, 2]
		self.coords[:,:,2] = self.screen[:, :, 0] * Matrix[2, 0] + self.screen[:, :, 1] * Matrix[2, 1] + self.screen[:, :, 2] * Matrix[2, 2]
		self.coords = parallelNormalizeVector(self.coords)

	def __init__(self, theta, phi, fov, resolution, x, y, z):
		self.origin = np.array([x, y, z], np.float32)
		Y, X = np.mgrid[:resolution[0], :resolution[1]]
		coords = np.zeros((resolution[0], resolution[1], 3), np.float64)
		width, height = resolution[0], resolution[1]
		X = (X - ((X.shape[1] - 1) / 2))
		Y = (Y - ((Y.shape[0] - 1) / 2))
		coords[:, :, 0] = X[:, :]
		coords[:, :, 1] = Y[:, :]
		coords[:, :, 2] = np.array(spherical2Cartesian(fov / 2, fov / 2, np.array(cartesian2Spherical(X[0, -1], Y[-1, 0], math.sqrt(((width / 2) ** 2 + (height / 2) ** 2))), np.float32)[2]), np.float32)[2]
		self.screen = coords * 1
		coords = parallelNormalizeVector(coords)
		self.coords = coords * 1
		self.theta = 0
		self.phi = 0
		self.rotate(theta, phi)
		self.MaxRenderDistance = 16 * math.pi

	def DoShading(self, Coords, normals):
		Vect = self.coords * 1

		normals = parallelNormalizeVector(normals)
		bright = np.sum(normals * Vect, axis = -1)
		MaxD = self.MaxRenderDistance
		distances = np.sqrt(np.sum(np.power(Coords - self.origin,2), axis=-1))
		distances[distances < 1] = 1
		distances[distances > MaxD] = MaxD
		distances = distances / MaxD
		normals[:,:,0] *= bright
		normals[:,:,1] *= bright
		normals[:,:,2] *= bright
		normals -= np.min(normals)
		frame = normals * 0
		frame[:,:,0] = normals[:,:,0] * (1 - distances) + np.max(normals) * (distances) * 0.3
		frame[:,:,1] = normals[:,:,1] * (1 - distances) + np.max(normals) * (distances) * 0.3
		frame[:,:,2] = normals[:,:,2] * (1 - distances) + np.max(normals) * (distances) * 0.3
		return frame

	def RenderFourierSeries(self, Coefficients, Thresh, bounds = None):
		Vect = np.reshape(self.coords * 1, (self.coords.shape[0] * self.coords.shape[1], 3))
		Coords = Vect * 0
		Coords[:, 0] = self.origin[0]
		Coords[:, 1] = self.origin[1]
		Coords[:, 2] = self.origin[2]

		T, DT = Newton.ModifiedNewtonSeries(Vect, Coords, self.origin * 1, Coefficients, thresh = Thresh, MaxDistance = self.MaxRenderDistance, bounds = bounds)
		# ~ T, DT = Newton.ClassicalNewtonSeries(Vect, Estimates, self.origin, thresh = Thresh)

		T = np.reshape(T, self.coords.shape)
		DT = np.reshape(DT, self.coords.shape)
		frame = self.DoShading(T, DT)
		return normalize(frame)

	# Func needs to have:
	#	bounds = ((xLower, xUpper), (yLower, yUpper), (zLower, zUpper))
	#	eval(coords); coords = numpy array dtype = np.float64 and shape = (3)
	#	ParallelEval(coords); coords = numpy array dtype = np.float64 and shape = (:, 3)
	def RenderExplicit(self, Func, Thresh, iterations = 500):
		Vect = np.reshape(self.coords * 1, (self.coords.shape[0] * self.coords.shape[1], 3))
		Coords = Vect * 0
		Coords[:, 0] = self.origin[0]
		Coords[:, 1] = self.origin[1]
		Coords[:, 2] = self.origin[2]

		T, DT = Newton.ModifiedNewtonExplicit(Vect, Coords, self.origin * 1, Func, iterations=iterations, thresh = Thresh, MaxDistance = self.MaxRenderDistance)
		# ~ T, DT = Newton.ClassicalNewtonTransform(Vect, Estimates, self.origin, Data, thresh = Thresh)

		T = np.reshape(T, self.coords.shape)
		DT = np.reshape(DT, self.coords.shape)
		frame = self.DoShading(T, DT)
		return normalize(frame)

