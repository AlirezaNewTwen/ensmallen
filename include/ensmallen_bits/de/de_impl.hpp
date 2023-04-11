/**
 * @file de_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of Differential Evolution an evolutionary algorithm used for
 * global optimization of arbitrary functions.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_DE_DE_IMPL_HPP
#define ENSMALLEN_DE_DE_IMPL_HPP

#include "de.hpp"
#include <queue>
#include <omp.h>

namespace ens {

inline DE::DE(const size_t populationSize ,
              const size_t maxGenerations,
              const double crossoverRate,
              const double differentialWeight,
              const double tolerance):
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverRate(crossoverRate),
    differentialWeight(differentialWeight),
    tolerance(tolerance)
{ /* Nothing to do here. */ }

//!Optimize the function
template<typename FunctionType,
         typename MatType,
         template<class> class VariablesConstraints,
         typename... CallbackTypes>
typename MatType::elem_type DE::Optimize(FunctionType& function,
                                         MatType& iterateIn,
                                         VariablesConstraints<MatType>& boxConstraint,
                                         CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Population matrix. Each column is a candidate.
  std::vector<BaseMatType> population;
  population.resize(populationSize);
  // Vector of fitness values corresponding to each candidate.
  arma::Col<ElemType> fitnessValues;

  // Make sure that we have the methods that we need.  Long name...
  traits::CheckArbitraryFunctionTypeAPI<
      FunctionType, BaseMatType>();
  RequireDenseFloatingPointType<BaseMatType>();

  // Population Size must be at least 3 for DE to work.
  if (populationSize < 3)
  {
    throw std::logic_error("CNE::Optimize(): population size should be at least"
        " 3!");
  }

  // Initialize helper variables.
  fitnessValues.set_size(populationSize);
  ElemType lastBestFitness = DBL_MAX;
  BaseMatType bestElement;
  std::queue<ElemType> performanceHorizon;
  int max_queue = 60;

  // Controls early termination of the optimization process.
  bool terminate = false;
  arma::arma_rng::set_seed_random();
  boxConstraint.RandomReseed(iterate);

  // Generate a population based on a Gaussian distribution around the given
  // starting point. Also finds the best element of the population.
  for (size_t i = 0; i < populationSize; i++)
  {
    //population[i].randn(iterate.n_rows, iterate.n_cols);
    //population[i] += iterate;

    //random displacement from initial seed within box constraints
    population[i] = (BaseMatType&)boxConstraint.RandomDisplacement(iterate);

    fitnessValues[i] = function.Evaluate(population[i]);

    Callback::Evaluate(*this, function, population[i], fitnessValues[i],
        callbacks...);

    if (fitnessValues[i] < lastBestFitness)
    {
      lastBestFitness = fitnessValues[i];
      bestElement = population[i];
    }
  }
  performanceHorizon.push(lastBestFitness); //append the best found

  // Iterate until maximum number of generations are completed.
  terminate |= Callback::BeginOptimization(*this, function, iterate,
      callbacks...);
  for (size_t gen = 0; gen < maxGenerations && !terminate; gen++)
  {
    // Generate new population based on /best/1/bin strategy.
    #pragma omp parallel for num_threads(8)
    for (int member = 0; member < populationSize; member++)
    {
      iterate = population[member];

      // Generate two different random numbers to choose two random members.
      size_t l = 0, m = 0;
      do
      {
        l = arma::randi<arma::uword>(arma::distr_param(0, populationSize - 1));
      }
      while (l == member);

      do
      {
        m = arma::randi<arma::uword>(arma::distr_param(0, populationSize - 1));
      }
      while (m == member && m == l);

      // Generate new "mutant" from two randomly chosen members.
      BaseMatType mutant = bestElement + differentialWeight *
          (population[l] - population[m]);

      //box constraints
      boxConstraint.RandBehave(mutant);

      // Perform crossover.
      const BaseMatType cr = arma::randu<BaseMatType>(iterate.n_rows);
      for (size_t it = 0; it < iterate.n_rows; it++)
      {
        if (cr[it] >= crossoverRate)
        {
          mutant[it] = iterate[it];
        }
      }

      ElemType iterateValue = function.Evaluate(iterate);
      #pragma omp critical 
      Callback::Evaluate(*this, function, iterate, iterateValue, callbacks...);

      const ElemType mutantValue = function.Evaluate(mutant);
      #pragma omp critical 
      Callback::Evaluate(*this, function, mutant, mutantValue, callbacks...);

      // Replace the current member if mutant is better.
      if (mutantValue < iterateValue)
      {
        iterate = mutant;
        iterateValue = mutantValue;
        #pragma omp critical 
        terminate |= Callback::StepTaken(*this, function, iterate,callbacks...);
      }

      fitnessValues[member] = iterateValue;
      population[member] = iterate;

    }

    // Append bestFitness to performanceHorizon.
    size_t id_min = fitnessValues.index_min();
    ElemType min_objective = fitnessValues[id_min];

    performanceHorizon.push(min_objective);
    while (performanceHorizon.size() > max_queue)
        performanceHorizon.pop();

    // Check for termination criteria.
    ElemType delta_obj_horizon = std::abs(performanceHorizon.front() - performanceHorizon.back());

    if (std::abs(lastBestFitness - min_objective) < tolerance && delta_obj_horizon < tolerance && gen>2)
    {
      Info << "DE: minimized within tolerance " << tolerance << "; "
          << "terminating optimization." << std::endl;
      lastBestFitness = min_objective;
      break;
    }

    // Update helper variables.
    lastBestFitness = min_objective;
    bestElement = population[id_min];

  }

  iterate = bestElement;

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return lastBestFitness;
}

} // namespace ens

#endif
