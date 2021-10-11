# rootFinder
A small python library for finding roots using numeric methods.

1 Import the archive:
  import rootFunctions as rootF
  
2 Define your function:
  def f(x):
    return x**2 + 2*x - 2
    
3 Define your derivative function(if needed):
  def f_diff(x):
    return 2*x + 2

4 Use the methods! An example:
  starting_x = 0.1
  epsilon_x = 1e-4
  epsilon_y = 1e-5
  rootF.newton_method(f, f_diff, starting_x, epsilon_x, epsilon_y, info=True)
  
