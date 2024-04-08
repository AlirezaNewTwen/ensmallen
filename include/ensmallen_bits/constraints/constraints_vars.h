#pragma once

#ifndef CONSTRAINT_VARS_HPP
#define CONSTRAINT_VARS_HPP

#include <armadillo>


namespace ens {


	template<typename MatType>
	class VariablesConstraints
	{
	typedef typename MatType::elem_type ElemType;
	typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

	private:
		arma::Col<ElemType> lowerbounds;
		arma::Col<ElemType> upperbounds;

	public:
		void set_box(arma::Col<ElemType>& lowerbound, arma::Col<ElemType>& upperbound) {
			lowerbounds = lowerbound;  upperbounds = upperbound;
		}

		constexpr arma::Col<ElemType>& get_lower_bound() const{ return lowerbounds; }
		constexpr arma::Col<ElemType>& get_upper_bound() const { return upperbounds; }

		VariablesConstraints() {};

		template<typename dim_space>
		VariablesConstraints(dim_space cube_dimension, ElemType lowerbound, ElemType upperbound) {
			lowerbounds = arma::Col<ElemType>(cube_dimension,arma::fill::value(lowerbound));
			upperbounds = arma::Col<ElemType>(cube_dimension, arma::fill::value(upperbound));
		}
		VariablesConstraints(arma::Col<ElemType>& lowerbound, arma::Col<ElemType>& upperbound) {
			lowerbounds = lowerbound;  upperbounds = upperbound;
		};

		inline void Stick2box(BaseMatType& population)
		{
			if (lowerbounds.is_empty()) return;
			for (size_t ii = 0; ii < population.n_rows; ii++)
			{
				population.row(ii).clamp(lowerbounds(ii), upperbounds(ii));
			}
		}

		inline void RandomReseed(BaseMatType& population)
		{
			if (lowerbounds.is_empty()) return;
			for (size_t ii = 0; ii < population.n_rows; ii++)
			{
				for (size_t jj = 0; jj < population.n_cols; jj++)
				{
					if (population(ii, jj) < lowerbounds(ii) || population(ii, jj) > upperbounds(ii))
					{
						auto rnd = arma::randu();
						population(ii, jj) = lowerbounds[ii] + (ElemType)rnd * (upperbounds(ii) - lowerbounds(ii));
					}
				}
			}
		}

		inline void Bounce(BaseMatType& population)
		{
			if (lowerbounds.is_empty()) return;
			for (size_t ii = 0; ii < population.n_rows; ii++)
			{
				for (size_t jj = 0; jj < population.n_cols; jj++)
				{
					if (population(ii, jj) < lowerbounds(ii))
					{
						ElemType infeasible_length = (lowerbounds(ii) - population(ii, jj))/(upperbounds(ii) - lowerbounds(ii));
						population(ii, jj) = lowerbounds(ii) + std::min(infeasible_length,(ElemType)1) * (upperbounds(ii) - lowerbounds(ii));
					}
					if (population(ii, jj) > upperbounds(ii))
					{
						ElemType infeasible_length = (population(ii, jj)- upperbounds(ii)) / (upperbounds(ii) - lowerbounds(ii));
						population(ii, jj) = lowerbounds(ii) + (1-std::min(infeasible_length, (ElemType)1)) * (upperbounds(ii) - lowerbounds(ii));
					}
				}
			}
		}

		inline BaseMatType RandomDisplacement(const BaseMatType& seed)
		{
			BaseMatType population(seed.n_rows, seed.n_cols);
			population = seed + arma::randn(seed.n_rows, seed.n_cols);
			RandomReseed(population);
			return population;
		}

		inline void RandBehave(BaseMatType& population)
		{
			if (lowerbounds.is_empty()) return;
			int seed = arma::randi<int>(arma::distr_param(0, 2));
			switch (seed)
			{
			case 0:
				Stick2box(population);
				break;
			case 1:
				Bounce(population);
				break;
			case 2:
			default:
				RandomReseed(population);
				break;
			}
		}
	};
}

#endif