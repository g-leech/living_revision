# Inefficient but interesting pi simulator. n > 50m plz.
# Pi is this quadrant area estimate * 4 quadrants
#@numerical_lulz
def pi(n):  
	hits = sum([ _unit_monte_carlo() for i in range(1, n+1) ])
	return 4 * hits / n

# Take the top-right quadrant of the unit circle. 
# Sample the quadrant; increment if it lands in the circle sector (~79% chance)
def _unit_monte_carlo():
	x = random() * 2 - 1
	y = random() * 2 - 1 
	return 1 if (x**2 + y**2 < 1) else 0
