# rootFinder<br />
A small python library for finding roots using numeric methods.

1 Import the archive:<br />
&nbsp;&nbsp;&nbsp;&nbsp;import rootFunctions as rootF<br />
  
2 Define your function:<br />
&nbsp;&nbsp;&nbsp;&nbsp;def f(x):<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return x**2 + 2*x - 2
    
3 Define your derivative function(if needed):<br />
&nbsp;&nbsp;&nbsp;&nbsp;def f_diff(x):<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return 2*x + 2<br />

4 Use the methods! An example:<br />
&nbsp;&nbsp;&nbsp;&nbsp;starting_x = 0.1<br />
&nbsp;&nbsp;&nbsp;&nbsp;epsilon_x = 1e-4<br />
&nbsp;&nbsp;&nbsp;&nbsp;epsilon_y = 1e-5<br />
&nbsp;&nbsp;&nbsp;&nbsp;rootF.newton_method(f, f_diff, starting_x, epsilon_x, epsilon_y, info=True)
  
