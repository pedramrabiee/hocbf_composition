Higher-Order Barrier Function Composition Library
This library provides a set of classes and utilities for working with barrier functions and higher-order barrier functions (HOCBFs) in the context of control systems and optimization problems.

Key Classes
Barrier
The Barrier class is the core of this library. It allows you to:

Assign a barrier function and its relative degree.
Assign the system dynamics to the barrier.
Compute the barrier function, the highest-order barrier function (HOCBF), and their Lie derivatives.
Retrieve information about the barrier, such as the relative degree and the list of barrier functions.
Methods
assign(barrier_func, rel_deg=1, alphas=None): Assigns the barrier function and its relative degree.
assign_dynamics(dynamics): Assigns the system dynamics to the barrier and generates the higher-order barrier functions.
h(x): Computes the barrier function for the given state x.
hocbf(x): Computes the highest-order barrier function for the given state x.
Lf_hocbf(x): Computes the Lie derivative of the HOCBF with respect to the system dynamics f.
Lg_hocbf(x): Computes the Lie derivative of the HOCBF with respect to the system dynamics g.
CompositionBarrier
The CompositionBarrier class is a subclass of Barrier that allows you to compose multiple barriers using a specific composition rule.

Methods
assign_barriers_and_rule(barriers, rule): Assigns a list of Barrier objects and a composition rule to the CompositionBarrier object.
h(x): Evaluates the barrier functions at given x.
hocbf(x): Computes the composed highest-order barrier function for the given state x.
SoftCompositionBarrier
The SoftCompositionBarrier class is a subclass of CompositionBarrier that implements soft composition rules, such as the union and intersection operations using softmax and softmin functions.

NonSmoothCompositionBarrier
The NonSmoothCompositionBarrier class is a subclass of CompositionBarrier that implements non-smooth composition rules, such as the union and intersection operations using the maximum and minimum functions.

Utility Functions
make_barrier_from_barrier(barrier, rel_deg=1): Constructs a new Barrier object using the highest-order barrier of the input Barrier object and assigns the same dynamics. The existing barriers are also appended to the new barrier.
Usage Example
python


Copy code
# Create a Barrier object
barrier = Barrier().assign(barrier_func=h, rel_deg=2, alphas=[alpha1, alpha2])
barrier.assign_dynamics(dynamics)

# Compute the barrier function, HOCBF, and their Lie derivatives
barrier_value = barrier.h(x)
hocbf_value = barrier.hocbf(x)
Lf_hocbf = barrier.Lf_hocbf(x)
Lg_hocbf = barrier.Lg_hocbf(x)

# Create a CompositionBarrier object
barriers = [barrier1, barrier2, barrier3]
composition_barrier = SoftCompositionBarrier().assign_barriers_and_rule(barriers, 'union')

# Compute the composed barrier function and HOCBF
composed_barrier_value = composition_barrier.h(x)
composed_hocbf_value = composition_barrier.hocbf(x)
