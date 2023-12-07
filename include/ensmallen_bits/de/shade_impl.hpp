/**
 * @file shade_impl.hpp
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
#ifndef ENSMALLEN_SHADE_SHADE_IMPL_HPP
#define ENSMALLEN_SHADE_SHADE_IMPL_HPP

#include "shade.hpp"
#include <queue>
#include <omp.h>
#include <unordered_set>

namespace ens {

inline SHADE::SHADE(const size_t populationSize ,
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
typename MatType::elem_type SHADE::Optimize(FunctionType& function,
                                         MatType& iterateIn,
                                         VariablesConstraints<MatType>& boxConstraint,
                                         CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  //parameter settings for SHADE SHADE SHADE SHADE SHADE SHADE
  double p_best_rate = 0.1;
  double arc_rate = 2.0;
  int problem_size = iterate.n_rows;
  int memory_size = problem_size;
  const int& pop_size = populationSize;
  // Initialize memory_sf, memory_cr, and memory_pos
  arma::vec memory_sf = 0.5 * arma::ones<arma::vec>(memory_size);
  arma::vec memory_cr = 0.5 * arma::ones<arma::vec>(memory_size);
  int memory_pos = 1;
  // Initialize archive
  int archive_NP = static_cast<int>(arc_rate * pop_size);
  std::vector<BaseMatType> archive_pop(0, BaseMatType(problem_size, 0.0));
  arma::Col<ElemType>      archive_funvalues = arma::zeros<arma::vec>(0);
  //SHADE SHADE SHADE SHADE SHADE SHADE

  // Population matrix. Each column is a candidate.
  std::vector<BaseMatType> population;
  population.resize(populationSize);
  // Vector of fitness values corresponding to each candidate.
  arma::Col<ElemType> fitnessValues;
  fitnessValues.set_size(populationSize);


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
  arma::Col<ElemType> fitnessDif;
  fitnessDif.set_size(populationSize);
  ElemType lastBestFitness = DBL_MAX;
  BaseMatType bestElement;
  std::queue<ElemType> performanceHorizon;
  int max_queue = 60;

  // Controls early termination of the optimization process.
  bool terminate = false;
  arma::arma_rng::set_seed_random();
  boxConstraint.RandomReseed(iterate);//why just tje iterate ALIREZA CHECK
  std::cout << "Need To be cheked ALIREZA";
  // Generate a population based on a Gaussian distribution around the given
  // starting point. Also finds the best element of the population.
  for (size_t i = 0; i < populationSize; i++)
  {
    //population[i].randn(iterate.n_rows, iterate.n_cols);
    //population[i] += iterate;
      std::cout << "Need To be cheked ALIREZA";
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

  	//For generating crossover rate SHADE SHADE SHADE SHADE SHADE SHADE
    std::cout << "Alireza check if fitnessValues is updetaed correctly during the loop";
    arma::umat goodChildren = arma::zeros<arma::umat>(populationSize);
      arma::uvec sorted_index = arma::sort_index(fitnessValues);
      // Generate random indices
      arma::vec mem_rand_index = arma::floor(arma::randu<arma::vec>(pop_size) * memory_size);
      arma::vec mu_sf = memory_sf(arma::conv_to<arma::uvec>::from(mem_rand_index));
      arma::vec mu_cr = memory_cr(arma::conv_to<arma::uvec>::from(mem_rand_index));
      // Generate crossover rate
      arma::vec crS = mu_cr + 0.1 * arma::randn<arma::vec>(pop_size);
      //arma::uvec term_pos = arma::find(mu_cr == -1);
      //crS.elem(term_pos).fill(0);
      crS.clamp(0.0, 1.0);

      // Generate scaling factor
      arma::vec sf = mu_sf + 0.1 * arma::tan(arma::datum::pi * (arma::randu<arma::vec>(pop_size) - 0.5));
      arma::uvec pos = arma::find(sf <= 0);
      // Regenerate sf for positions where sf is less than or equal to 0
      while (!pos.is_empty()) {
          sf.elem(pos) = mu_sf.elem(pos) + 0.1 * arma::tan(arma::datum::pi * (arma::randu<arma::vec>(pos.n_elem) - 0.5));
          pos = arma::find(sf <= 0);
      }
      sf = arma::min(sf, 1.0);


      std::vector<BaseMatType> popAll = population;
      popAll.insert(popAll.end(), archive_pop.begin(), archive_pop.end());

      //// Generate r1 and r2
      arma::uvec r0 = arma::linspace<arma::uvec>(0, pop_size-1, pop_size);
      arma::uvec r1(pop_size);
      arma::uvec r2(pop_size);
      gnR1R2(pop_size, popAll.size(), r0, r1, r2);

      //generate a vector of best solutions
      // Calculate the number of best solutions to choose (at least two best solutions)
      int pNP = std::max(static_cast<int>(std::round(p_best_rate * pop_size)), 2);
      arma::uvec randindex = arma::randi<arma::uvec>(pop_size, arma::distr_param(0, pNP - 1));

      std::vector<BaseMatType> pbest(pop_size);
	  //arma::vec pbestFitness = arma::zeros<arma::vec>(pop_size);
      for (std::size_t i = 0; i < pop_size; ++i) {
          //std::size_t index = randindex[i];
          if (randindex[i] < sorted_index.size()) {
              pbest[i] = population[sorted_index[randindex[i]]];
              //pbestFitness[i] = fitnessValues[sorted_index[randindex[i]]];
          }
      }

    std::vector<BaseMatType> vi(pop_size, BaseMatType(problem_size, 0.0));
    for (int i = 0; i < pop_size; ++i) {
		// Generate mutant vector

        vi[i] = population[i] + sf[i] * (pbest[i]-population[i]+population[r1[i]]- popAll[r2[i]]);
		//box constraints
		//boxConstraint.RandBehave(Vi[i]);
	}
    arma::mat randMask = arma::randu(pop_size, problem_size);
    arma::mat crReplicated = arma::repmat(crS, 1, problem_size);

    arma::umat crMask = randMask > crReplicated;
    arma::uvec jrand = arma::randi<arma::uvec>(pop_size, arma::distr_param(0, problem_size - 1));

    for (arma::uword i = 0; i < pop_size; ++i) {
        crMask(i, jrand(i)) = 0;
    }


    std::vector<BaseMatType> ui=vi;

	for (arma::uword i = 0; i < pop_size; ++i) {
		ui[i].elem(arma::find(crMask.row(i)))=population[i].elem(arma::find(crMask.row(i)));
	}


  	//std::vector<BaseMatType> pbest=population.rows(sorted_index.elem(randindex));

  	//SHADE SHADE SHADE SHADE SHADE SHADE

    // Archiving data SHADE SHADE SHADE SHADE SHADE SHADE
    std::vector<BaseMatType> popToArchive(0, BaseMatType(problem_size, 0.0));
    arma::Col<ElemType> funvalueToArchive;
    // Generate new population based on /best/1/bin strategy.
    #pragma omp parallel for num_threads(8)
    for (int member = 0; member < populationSize; member++)
    {
      iterate = population[member];

      
      // Generate new "mutant" from two randomly chosen members.
      BaseMatType mutant = ui[member];

      //box constraints
      boxConstraint.RandBehave(mutant);


      ElemType iterateValue = function.Evaluate(iterate);
      #pragma omp critical 
      Callback::Evaluate(*this, function, iterate, iterateValue, callbacks...);

      const ElemType mutantValue = function.Evaluate(mutant);
      #pragma omp critical 
      Callback::Evaluate(*this, function, mutant, mutantValue, callbacks...);
      //fitnessDif[member] = abs(mutantValue - iterateValue);
      fitnessDif[member] = iterateValue - mutantValue;
      // Replace the current member if mutant is better.
      if (mutantValue < iterateValue)
      {
        iterate = mutant;
        iterateValue = mutantValue;
        goodChildren(member) = true;
        //goodF[member] = true;
        #pragma omp critical 
        terminate |= Callback::StepTaken(*this, function, iterate,callbacks...);
      }
      else
      {
          popToArchive.push_back(mutant);
      	  funvalueToArchive.insert_rows(funvalueToArchive.n_rows, mutantValue);
      }

      fitnessValues[member] = iterateValue;
      population[member] = iterate;

    }
    arma::vec goodCr = crS.elem(arma::find(goodChildren));
    arma::vec goodSf = sf.elem(arma::find(goodChildren));
    arma::vec dif_val = fitnessDif.elem(arma::find(goodChildren));

  	//archiveSHADEUPDATEUPDATEUPDATEUPDATEUPDATEUPDATEUPDATEUPDATEUPDATEUPDATEU
    std::vector<BaseMatType> tempArcive = archive_pop;
    tempArcive.insert(tempArcive.end(), popToArchive.begin(), popToArchive.end());
    arma::Col<ElemType> tempfunvalues = arma::join_cols(archive_funvalues, funvalueToArchive);
    //tempfunvalues.insert_rows(tempfunvalues.n_rows, funvalueToArchive);

    std::cout<< "Aliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"<<std::endl;

    for (int i = 0; i < archive_pop.size(); ++i)
    {
        std::cout << "archive_pop." << i << "=" << archive_pop[i] << std::endl;
    }
    for (int i = 0; i < popToArchive.size(); ++i)
    {
        std::cout << "popToArchive." << i << "=" << popToArchive[i] << std::endl;
    }
    for (int i = 0; i < tempArcive.size(); ++i)
    {
        std::cout << "tempArcive." << i << "=" << tempArcive[i] << std::endl;
    }

    std::unordered_set<std::string> uniqueIdentifiers;
    std::vector<bool> hasDuplicates(tempArcive.size(), false);

    for (std::size_t i = 0; i < tempArcive.size(); ++i) {
        const BaseMatType& element = tempArcive[i];

        // Convert BaseMatType to a unique identifier (string ).
        std::ostringstream oss;
        oss << element;  // You might want to use a more appropriate serialization method based on your needs.
        std::string identifier = oss.str();

        // Check for duplicates.
        if (!uniqueIdentifiers.insert(identifier).second) {
            hasDuplicates[i] = true;  // Set the corresponding flag to true.
            if(i<archive_pop.size())
            {
                throw std::logic_error("SHADE::Optimize(): there is a identical member in SHADE Archive");
			}
        }
        else
        {
            if (i >= archive_pop.size() && archive_pop.size() < archive_NP)
            {
                archive_pop.push_back(element);
				archive_funvalues.insert_rows(archive_funvalues.n_rows, tempfunvalues[i]);
			}
            else if (i >= archive_pop.size() && archive_pop.size() == archive_NP)
            {
	            int randIndex = arma::randi<arma::uvec>(1, arma::distr_param(0, archive_NP - 1))[0];
				archive_pop[randIndex] = element;
                archive_funvalues[randIndex] = tempfunvalues[i];
            }
            else if (archive_pop.size() > archive_NP)
            {
	            throw std::logic_error("SHADE::Optimize(): Archive exceeds the maximum size");
            }
        }
    }


  //  for (const BaseMatType& element : tempArcive)
  //  {
  //  	std::ostringstream oss;
		//oss << element;  // You might want to use a more appropriate serialization method based on your needs.
		//std::string identifier = oss.str();
		//// Check for duplicates.
		//if (!uniqueIdentifiers.insert(identifier).second)
		//{
  //          archive_pop.push_back(element);
		//}

  //  }

    	//archiveSHADEUPDATEUPDATEUPDATE
    // Append bestFitness to performanceHorizon.
    size_t id_min = fitnessValues.index_min();
    ElemType min_objective = fitnessValues[id_min];

    if (numel(goodCr) > 0) {
        double sum_dif = arma::sum(dif_val);
        dif_val /= sum_dif;

        // Update the memory of scaling factor
        memory_sf(memory_pos - 1) = arma::dot(dif_val, goodSf % goodSf) / arma::dot(dif_val, goodSf);

        // Update the memory of crossover rate
        if (arma::max(goodCr) == 0 || memory_cr(memory_pos - 1) == -1) {
            memory_cr(memory_pos - 1) = -1;
        }
        else {
            memory_cr(memory_pos - 1) = arma::dot(dif_val, goodCr % goodCr) / arma::dot(dif_val, goodCr);
        }

        memory_pos++;
        if (memory_pos > memory_size) memory_pos = 1;
    }


    performanceHorizon.push(min_objective);
    while (performanceHorizon.size() > max_queue)
        performanceHorizon.pop();

    // Check for termination criteria.
    ElemType delta_obj_horizon = std::abs(performanceHorizon.front() - performanceHorizon.back());

    if (std::abs(lastBestFitness - min_objective) < tolerance && delta_obj_horizon < tolerance && gen>2)
    {
      Info << "SHADE: minimized within tolerance " << tolerance << "; "
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


inline void SHADE::gnR1R2(int NP1, int NP2, const arma::uvec& r0, arma::uvec& r1, arma::uvec& r2) {


    // Initialize r1

    r1 = arma::randi<arma::uvec>(NP1, arma::distr_param(0, NP1 - 1));

    // Regenerate r1 if it is equal to r0
    for (int i = 0; i < 99999999; ++i) {
        arma::uvec pos = (r1 == r0);
        if (arma::sum(pos) == 0) {
            break;
        }
        else {
            arma::uvec newValues = arma::randi<arma::uvec>(arma::sum(pos), arma::distr_param(0, NP1 - 1));
            //r1.elem(arma::find(pos)).fill(0);  // Fill with zeros to avoid conflicts
            r1.elem(arma::find(pos)) = newValues;
        }
        if (i > 1000) {
            throw std::runtime_error("Cannot generate r1 in 1000 iterations");
        }
    }

    // Initialize r2

    r2 = arma::randi<arma::uvec>(NP1, arma::distr_param(0, NP2 - 1));

    // Regenerate r2 if it is equal to r0 or r1
    for (int i = 0; i < 99999999; ++i) {
        arma::uvec pos = ((r2 == r1) || (r2 == r0));
        if (arma::sum(pos) == 0) {
            break;
        }
        else {
            arma::uvec newValues = arma::randi<arma::uvec>(arma::sum(pos), arma::distr_param(0, NP2 - 1));
            //r2.elem(arma::find(pos)).fill(0);  // Fill with zeros to avoid conflicts
            r2.elem(arma::find(pos)) = newValues;
        }
        if (i > 1000) {
            throw std::runtime_error("Cannot generate r2 in 1000 iterations");
        }
    }

}

struct Archive {
    arma::mat pop;
    arma::mat funvalues;
    size_t NP; // Assuming NP is the archive size
};

inline Archive updateArchive(const Archive& archive, const arma::mat& pop, const arma::mat& funvalue) {
    // Check if the archive size is zero, then return
    if (archive.NP == 0) {
        return archive;
    }

    // Check if the dimensions of pop and funvalue are consistent
    if (pop.n_rows != funvalue.n_rows) {
        throw std::runtime_error("Inconsistent dimensions between pop and funvalue.");
    }

    // Method 2: Remove duplicate elements
    arma::mat popAll = arma::join_cols(archive.pop, pop);
    arma::mat funvalues = arma::join_cols(archive.funvalues, funvalue);

    //arma::uvec IX;
    //arma::uvec dummy; // Not used, as unique returns values
    //arma::unique(dummy, IX, popAll, "rows");
    arma::uvec IX = arma::find(arma::diff(popAll, 1, 0) != 0);


    // If there are duplicates, keep unique solutions
    if (IX.n_elem < popAll.n_rows) {
        popAll = popAll.rows(IX);
        funvalues = funvalues.rows(IX);
    }

    Archive updatedArchive;

    // Check if the total number of solutions is less than or equal to archive.NP
    if (popAll.n_rows <= archive.NP) {
        // Add all new individuals to the archive
        updatedArchive.pop = popAll;
        updatedArchive.funvalues = funvalues;
    }
    else {
        // Randomly remove some solutions to maintain the archive size
        arma::uvec rndpos = arma::randi<arma::uvec>(archive.NP, arma::distr_param(0, popAll.n_rows - 1));
        updatedArchive.pop = popAll.rows(rndpos);
        updatedArchive.funvalues = funvalues.rows(rndpos);
    }

    updatedArchive.NP = archive.NP;
    return updatedArchive;
}

} // namespace ens

#endif
