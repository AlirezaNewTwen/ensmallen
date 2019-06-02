/**
 * @file svrg_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of stochastic variance reduced gradient (SVRG).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SVRG_SVRG_IMPL_HPP
#define ENSMALLEN_SVRG_SVRG_IMPL_HPP

// In case it hasn't been included yet.
#include "svrg.hpp"

namespace ens {

template<typename UpdatePolicyType, typename DecayPolicyType>
SVRGType<UpdatePolicyType, DecayPolicyType>::SVRGType(
    const double stepSize,
    const size_t batchSize,
    const size_t maxIterations,
    const size_t innerIterations,
    const double tolerance,
    const bool shuffle,
    const UpdatePolicyType& updatePolicy,
    const DecayPolicyType& decayPolicy,
    const bool resetPolicy) :
    stepSize(stepSize),
    batchSize(batchSize),
    maxIterations(maxIterations),
    innerIterations(innerIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    updatePolicy(updatePolicy),
    decayPolicy(decayPolicy),
    resetPolicy(resetPolicy)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename UpdatePolicyType, typename DecayPolicyType>
template<typename DecomposableFunctionType, typename MatType, typename GradType>
typename MatType::elem_type SVRGType<UpdatePolicyType,
                                     DecayPolicyType>::Optimize(
    DecomposableFunctionType& functionIn,
    MatType& iterateIn)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  typedef Function<DecomposableFunctionType, BaseMatType, BaseGradType,
      decltype(this)> FullFunctionType;
  FullFunctionType& function(static_cast<FullFunctionType&>(functionIn));

  traits::CheckDecomposableFunctionTypeAPI<DecomposableFunctionType,
      BaseMatType, BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  typedef typename UpdatePolicyType::template Policy<BaseMatType, BaseGradType>
      InstUpdatePolicyType;
  typedef typename DecayPolicyType::template Policy<BaseMatType, BaseGradType>
      InstDecayPolicyType;

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  ElemType overallObjective = 0;
  ElemType lastObjective = DBL_MAX;

  // Set epoch length to n / b if the user asked for.
  if (innerIterations == 0)
    innerIterations = numFunctions;

  // Initialize the decay policy.
  if (!isInitialized ||
      !instDecayPolicy.Has<InstDecayPolicyType>())
  {
    instDecayPolicy.Clean();
    instDecayPolicy.Set<InstDecayPolicyType>(
        new InstDecayPolicyType(decayPolicy));
  }

  // Initialize the update policy.
  if (resetPolicy || !isInitialized ||
      !instUpdatePolicy.Has<InstUpdatePolicyType>())
  {
    instUpdatePolicy.Clean();
    instUpdatePolicy.Set<InstUpdatePolicyType>(
        new InstUpdatePolicyType(updatePolicy, iterate.n_rows, iterate.n_cols));
    isInitialized = true;
  }

  // Now iterate!
  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  BaseGradType gradient0(iterate.n_rows, iterate.n_cols);
  BaseMatType iterate0;

  // Find the number of batches.
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; ++i)
  {
    // Calculate the objective function.
    overallObjective = 0;
    for (size_t f = 0; f < numFunctions; f += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
      overallObjective += function.Evaluate(iterate, f, effectiveBatchSize);
    }

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "SVRG: converged to " << overallObjective
          << "; terminating  with failure.  Try a smaller step size?"
          << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Info << "SVRG: minimized within tolerance " << tolerance
          << "; terminating optimization." << std::endl;
      return overallObjective;
    }

    lastObjective = overallObjective;

    // Compute the full gradient.
    size_t effectiveBatchSize = std::min(batchSize, numFunctions);
    BaseGradType fullGradient(iterate.n_rows, iterate.n_cols);
    function.Gradient(iterate, 0, fullGradient, effectiveBatchSize);
    for (size_t f = effectiveBatchSize; f < numFunctions;
        /* incrementing done manually */)
    {
      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - f);

      function.Gradient(iterate, f, gradient, effectiveBatchSize);
      fullGradient += gradient;

      f += effectiveBatchSize;
    }
    fullGradient /= (double) numFunctions;

    // Store current parameter for the calculation of the variance reduced
    // gradient.
    iterate0 = iterate;

    for (size_t f = 0, currentFunction = 0; f < innerIterations;
        /* incrementing done manually */)
    {
      // Is this iteration the start of a sequence?
      if ((currentFunction % numFunctions) == 0)
      {
        currentFunction = 0;

        // Determine order of visitation.
        if (shuffle)
          function.Shuffle();
      }

      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - currentFunction);

      // Calculate variance reduced gradient.
      function.Gradient(iterate, currentFunction, gradient,
          effectiveBatchSize);
      function.Gradient(iterate0, currentFunction, gradient0,
          effectiveBatchSize);

      // Use the update policy to take a step.
      instUpdatePolicy.As<InstUpdatePolicyType>().Update(iterate, fullGradient,
          gradient, gradient0, effectiveBatchSize, stepSize);

      currentFunction += effectiveBatchSize;
      f += effectiveBatchSize;
    }

    // Update the learning rate if requested by the user.
    instDecayPolicy.As<InstDecayPolicyType>().Update(iterate, iterate0,
        gradient, fullGradient, numBatches, stepSize);
  }

  Info << "SVRG: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
    overallObjective += function.Evaluate(iterate, i, effectiveBatchSize);
  }
  return overallObjective;
}

} // namespace ens

#endif
