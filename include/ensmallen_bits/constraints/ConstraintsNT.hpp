#pragma once
/**
 * @file 111.hpp
 * @author Alirezza
 *
 * some limitation regarding the Inequality Constraint
 * each 
 */
#ifndef CONSTRAINT_NT_HPP
#define CONSTRAINT_NT_HPP
#include <armadillo>
#include "exprtk.hpp"
//#include "C:\Hexadrive\exprtk\exprtk.hpp"


namespace ens {
	enum ConstraintType { bounds, inequality, expression, noConstraint, equality};
	enum InequalityType { noConstrains,less_than, less_than_or_equal_to, greater_than, greater_than_or_equal_to, equal_to };

	template<typename MatType>
	class ConstraintsNT
	{
		typedef typename MatType::elem_type ElemType;
		typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
	private:
		std::vector<std::string> _parametherTags;
		std::shared_ptr<std::vector<double>> _initial_values=nullptr;
		arma::Col<ElemType> lowerbounds;
		arma::Col<ElemType> upperbounds;
		int dimProb;//dimension of problem
		ConstraintType type;
		std::map<ConstraintType, int> constraintCounts;
		std::vector<ConstraintType> typOfConstraints;
		std::vector<InequalityType> inqualityType;
		arma::Col<ElemType> threshold;
		arma::imat constrainedIdxInq;
		arma::imat constrainedIdxExpr;
		std::vector<exprtk::expression<ElemType>> constraintExpressions;
		std::vector<std::vector<int>> exprPrmtrIndcs;
		std::vector<ElemType> prmtrVector;
		std::vector<std::string> strExpressions;
		exprtk::symbol_table<ElemType> symbolTable;


	public:
		ConstraintsNT() {}

		template<typename dim_space>
		ConstraintsNT(dim_space cube_dimension, ElemType lowerbound, ElemType upperbound) :
			lowerbounds(cube_dimension, arma::fill::value(lowerbound)),
			upperbounds(cube_dimension, arma::fill::value(upperbound)) {}
		ConstraintsNT(arma::Col<ElemType>& lowerbound, arma::Col<ElemType>& upperbound) :
			lowerbounds(lowerbound),
			upperbounds(upperbound),
			type(ConstraintType::bounds),
			typOfConstraints(1,ConstraintType::bounds)
		{}

		//ConstraintsNT(arma::Col<std::string>& parametherTags, arma::Col<ElemType>& lowerbound, arma::Col<ElemType>& upperbound) :
		//	_parametherTags(parametherTags),
		//	lowerbounds(lowerbound),
		//	upperbounds(upperbound),
		//	type(ConstraintType::bounds),
		//	typOfConstraints(1, ConstraintType::bounds)
		//{}

		ConstraintsNT(const std::vector<std::tuple<std::string, double, double>>& uncertain_parameters_,const std::shared_ptr<std::vector<double>>& initailValues):
			_initial_values(initailValues)
		{
			dimProb= uncertain_parameters_.size();
			lowerbounds.set_size(dimProb);
			upperbounds.set_size(dimProb);
			_parametherTags.resize(dimProb);
			typOfConstraints.clear();
			typOfConstraints.push_back(ConstraintType::bounds);
			constraintCounts[ConstraintType::bounds] = dimProb;
			constraintCounts[ConstraintType::inequality] = 0;
			constraintCounts[ConstraintType::expression] = 0;
			//constraintCounts[ConstraintType::noConstraint] = 4;
			constraintCounts[ConstraintType::equality] =0;
			for (size_t i = 0; i < uncertain_parameters_.size(); i++)
			{
				_parametherTags[i] = std::get<0>(uncertain_parameters_[i]);
				lowerbounds(i) = std::get<1>(uncertain_parameters_[i]);
				upperbounds(i) = std::get<2>(uncertain_parameters_[i]);
			}


			std::set<std::string> uniqueValues(_parametherTags.begin(), _parametherTags.end());
			if (uniqueValues.size() != _parametherTags.size()) 
			{
				throw std::runtime_error("There are duplicate parameter tags.");
			}
		}


		ConstraintsNT(ConstraintType type_, arma::Col<ElemType>& lowerbound_, arma::Col<ElemType>& upperbound_,
			arma::imat& constrainedIdxInq, arma::Col<ElemType> threshold_) :
			lowerbounds(lowerbound_),
			upperbounds(upperbound_),
			type(type_),
			threshold(threshold_),
			constrainedIdxInq(constrainedIdxInq) {
		}

		ConstraintsNT(ConstraintType type_, arma::Col<ElemType>& lowerbound_, arma::Col<ElemType>& upperbound_,
			arma::imat& constrainedIdxInq, arma::Col<ElemType> threshold_, std::vector<InequalityType> inqtyp) :
			lowerbounds(lowerbound_),
			upperbounds(upperbound_),
			type(type_),
			threshold(threshold_),
			inqualityType(inqtyp),
			constrainedIdxInq(constrainedIdxInq) {
		}

		ConstraintsNT(ConstraintType type_,  arma::Col<ElemType>& lowerBounds,  arma::Col<ElemType>& upperBounds,
			 std::vector<std::string>& StrExpressions) :
			lowerbounds(lowerBounds),
			upperbounds(upperBounds),
			dimProb(lowerBounds.n_rows),
			type(type_),
			strExpressions(StrExpressions) {
			//compiling the expressions
			//initializing with a dummy vector
			prmtrVector = std::vector<ElemType>(lowerBounds.begin(), lowerBounds.end());
			symbolTable.add_vector("x", prmtrVector); // Dummy vector; will be updated during evaluation.
			symbolTable.add_constants();

			// Compile.
			for (const std::string& exprStr : strExpressions) {
				exprtk::expression<ElemType> compiledExpression;
				compiledExpression.register_symbol_table(symbolTable);

				if (!exprtk::parser<ElemType>().compile(exprStr, compiledExpression)) {
					throw std::runtime_error("Error compiling constraint expression.");
				}
				constraintExpressions.push_back(compiledExpression);


				// Find and store the parameter indices in the expression
				std::vector<int> indices;
				arma::irowvec indices1(dimProb, arma::fill::zeros);
				std::size_t found = exprStr.find("x[");
				while (found != std::string::npos) {
					std::size_t end = exprStr.find("]", found);
					if (end != std::string::npos) {
						std::string indexStr = exprStr.substr(found + 2, end - found - 2);
						int index = std::stoi(indexStr);
						if (index >= lowerBounds.n_elem) {
							throw std::runtime_error("Index in expression exceeds the size of parameter.");
						}
						if(std::find(indices.begin(), indices.end(), index) == indices.end())
						indices.push_back(index);
						indices1(index) = 1;
						found = exprStr.find("x[", end);
					}
				}
				exprPrmtrIndcs.push_back(indices);
				constrainedIdxExpr.insert_rows(constrainedIdxExpr.n_rows, indices1);
			}
		}


		

		std::vector<bool> checkValidity(const BaseMatType& solution, ConstraintType typOfConstraint) 
		{
			switch (typOfConstraint)
			{
			case ConstraintType::bounds:
				return checkBoundsConstraint(solution);

			case ConstraintType::inequality:
				return checkInequalityConstraints(solution);
			case ConstraintType::noConstraint:
			{
				std::vector<bool> isSatisfied(solution.n_elem, true);
				return isSatisfied;
			}
			case ConstraintType::expression:
				return checkExpressionsConstraints(solution);
			default:
				throw std::runtime_error("Invalid constraint type.");
			}

		}



		std::vector<bool> checkBoundsConstraint(const BaseMatType& solution) 
		{
			std::vector<bool> isSatisfied(solution.n_elem, true);
			for (size_t i=0; i < solution.n_elem; i++)
			{
				if (solution(i) < lowerbounds(i) || solution(i) > upperbounds(i))
				{
					isSatisfied[i] = false; // Constraint violated for this index.
				}
			}
			return isSatisfied;
		}



		std::vector<bool> checkInequalityConstraints(const BaseMatType& solution) 
		{
			std::vector<bool> isSatisfied(constrainedIdxInq.n_rows, true);
			// Ensure 
			if (threshold.n_elem != constrainedIdxInq.n_rows)
			{
				throw std::runtime_error("Threshold and constrained Indices dimensions do not match.");
			}
			if (constrainedIdxInq.n_cols != solution.n_rows)
			{
				throw std::runtime_error("constrained Indices and the dimension do not match.");
			}
			if (solution.n_cols != 1)
			{
				throw std::runtime_error("solution must be a column vector.");
			}
			for (size_t i = 0; i < constrainedIdxInq.n_rows; i++)
			{
				isSatisfied[i] = checkInequalityConstraint(solution, i);
			}

			return isSatisfied;
		}

		bool checkInequalityConstraint(const BaseMatType& solution, size_t inequalityIndex)
		{
			ElemType sum1 = dot(solution, constrainedIdxInq.row(inequalityIndex));
			switch (inqualityType[inequalityIndex])
			{
			case InequalityType::less_than:
				if (!(sum1 < threshold(inequalityIndex)))
				{
					return false; // Constraint violated for this index.
				}
				break;
			case InequalityType::less_than_or_equal_to:
				if (!(sum1 <= threshold(inequalityIndex)))
				{
					return false; // Constraint violated for this index.
				}
				break;
			case InequalityType::greater_than:
				if (!(sum1 > threshold(inequalityIndex)))
				{
					return false; // Constraint violated for this index.
				}
				break;
			case InequalityType::greater_than_or_equal_to:

				if (!(sum1 >= threshold(inequalityIndex)))
				{
					return false; // Constraint violated for this index.
				}
				break;
			case InequalityType::equal_to:
				if (!(sum1 == threshold(inequalityIndex)))
				{
					return false; // Constraint violated for this index.
				}
				break;
			default:
				throw std::runtime_error("Invalid inequality type.");
			}
			return true;
		}

		std::vector<bool> checkExpressionsConstraints(const BaseMatType& solution) 
		{
			std::vector<bool> isSatisfied(constraintExpressions.size(), true);
			std::vector<ElemType> prmtrVector1 = arma::conv_to<std::vector<ElemType>>::from(solution);
			prmtrVector = prmtrVector1;
			//std::cout << "Particle: " << std::endl << solution << std::endl;
			for (size_t i = 0; i < constraintExpressions.size(); i++)
			{
				//std::cout << "Expression " << i << ": " << strExpressions[i] << std::endl;
				//std::cout << "Parameter Indices: ";
				//for (int index : exprPrmtrIndcs[i])
				//{
				//	std::cout << index << " ";
				//}
				//std::cout << std::endl;
				//std::cout << "Result: " << constraintExpressions[i].value() << std::endl;
				//// check if the value is not bool, throw and warning
				if (! (constraintExpressions[i].value() == 0 || constraintExpressions[i].value() == 1))
				{
					std::cout << "Expression " << i << ": " << strExpressions[i] << std::endl;
					throw std::runtime_error("The expression needs to be logic");
				}
				isSatisfied[i] = constraintExpressions[i].value();
			}
			return isSatisfied;
		}

		bool checkExpressionConstraint(const BaseMatType& solution, size_t expressionIndex)
		{
			std::vector<ElemType> prmtrVector1 = arma::conv_to<std::vector<ElemType>>::from(solution);
			prmtrVector = prmtrVector1;
			return constraintExpressions[expressionIndex].value();
		}

		inline void enforceConstraints(BaseMatType& solution)
		{
			if(_initial_values==nullptr)
			{
				enforceConstraints(solution, typOfConstraints);
			}
			else
			{
				solution= solution - arma::conv_to<BaseMatType>::from(*_initial_values);
				enforceConstraints(solution, typOfConstraints);
				solution = solution + arma::conv_to<BaseMatType>::from(*_initial_values);
			}
			
		}


		void enforceConstraints(BaseMatType& solution, std::vector<ConstraintType> typsOfConstraints)
		{
			for (size_t i = 0; i < typsOfConstraints.size(); i++)
			{
				switch (typsOfConstraints[i])
				{
				case ConstraintType::bounds:
					{
						std::vector<bool> validity= checkBoundsConstraint(solution);
						if (std::any_of(validity.begin(), validity.end(), [](bool valid) { return !valid; })) 
						{
							applyBounds(solution, validity);
						}
					}
					break;
				case ConstraintType::inequality:
					{
						std::vector<bool> validity = checkInequalityConstraints(solution);
						if (std::any_of(validity.begin(), validity.end(), [](bool valid) { return !valid; }))
						{
							applyInequalities(solution, validity);
						}
					}
					break;
				case ConstraintType::noConstraint:
				{
					return;
				}
				case ConstraintType::expression:
					{
						std::vector<bool> validity = checkExpressionsConstraints(solution);
						if (std::any_of(validity.begin(), validity.end(), [](bool valid) { return !valid; }))
						{
							applyExpressions(solution, validity);
						}
					}
					break;
				default:
					throw std::runtime_error("Invalid constraint type.");
				}
			}


		}

		void applyBounds (BaseMatType& solution, std::vector<bool>& isSatisfied)
		{
			//std::vector<bool> isSatisfied = checkBoundsConstraint(solution);
			for (size_t i=0; i<isSatisfied .size(); i++)
			{
				if (isSatisfied[i] == true)
					continue;
				int seed = arma::randi<int>(arma::distr_param(0, 2));
				switch (seed)
				{
				case 0://Stick2box
					solution.row(i).clamp(lowerbounds(i), upperbounds(i));
					break;
				case 1://Bounce
					bounceValue(solution(i), i);
					break;
				case 2://Random Value
				default:
					randomValue(solution(i), i);
					break;
				}
			}
			isSatisfied = checkBoundsConstraint(solution);
			if (!std::all_of(isSatisfied.begin(), isSatisfied.end(), [](bool valid) { return valid; }))
			{
				throw std::runtime_error("Invalid validity found");
			}
		}

		void applyInequalities(BaseMatType& solution, std::vector<bool>& isSatisfied)
		{
			//std::vector<bool> isSatisfied = checkInequalityConstraints(solution);
			for (size_t ic = 0; ic < isSatisfied.size(); ic++) //loop over each inequality constrains 
			{
				if (isSatisfied[ic] == true)
					continue;
				if (checkInequalityConstraint(solution, ic))
					continue;
				BaseMatType  tmpSol = solution;

				arma::uvec nonZeroIndices = arma::find(constrainedIdxInq.row(ic));

				//first try to change the first parameter that is included in the constrain
				//the first parameter in the constraint is considered the most important one
				int index = nonZeroIndices(0);
				arma::vec rnd = arma::randu(20);
				arma::uvec randIndext = arma::randperm(20);
				for (size_t j = 0; j < 20; j++) 
				{
					tmpSol[index] = rnd[randIndext(j)] * (upperbounds(index) - lowerbounds(index)) + lowerbounds(index);
					if (checkInequalityConstraint(tmpSol, ic))
					{
						solution = tmpSol;
						isSatisfied[ic]= true;
						break;
					}
				}

				if (isSatisfied[ic])
					continue;
				tmpSol = solution;

				//Second try to change the other parameters that are included ONLY in this constrain
				arma::uvec parameterToChange = nonZeroIndices(arma::randperm(nonZeroIndices.n_elem));
				arma::uvec action = arma::randperm(3);
				std::vector<int> repeatedParameter;
				for (size_t j = 0; j < parameterToChange.n_elem; j++) //loop over parameter to be change 
				{
					index = parameterToChange(j);
					if (index == nonZeroIndices(0))
						continue;
					//check if the parameter is included in other constrains
					arma::uvec tmp = arma::find(constrainedIdxInq.col(index));
					if (tmp.n_elem > 1)
					{
						repeatedParameter.push_back(index);
						continue;
					}
					//change the parameter
					arma::vec rnd = arma::randu(20);
					for (size_t j = 0; j < rnd.n_elem; j++)
					{
						tmpSol[index] = rnd[j] * (upperbounds(index) - lowerbounds(index)) + lowerbounds(index);
						if (checkInequalityConstraint(tmpSol, ic))
						{
							solution = tmpSol;
							isSatisfied[ic] = true;
							break;
						}
					}
				}

				if (isSatisfied[ic])
					continue;
				tmpSol = solution;

				//third, change the other parameters that are included in several constrains 
				for (size_t j = 0; j < repeatedParameter.size(); j++) //loop over parameter that are included in several constrains 
				{
					index = repeatedParameter[j];
					if (index == nonZeroIndices(0))
						continue;
					//change the parameter
					arma::vec rnd = arma::randu(20);
					for (size_t j = 0; j < rnd.n_elem; j++)
					{
						tmpSol[index] = rnd[j] * (upperbounds(index) - lowerbounds(index)) + lowerbounds(index);
						if (checkInequalityConstraint(tmpSol, ic))
						{
							solution = tmpSol;
							isSatisfied[ic] = true;
							break;
						}
					}
					if (isSatisfied[ic])
						break;
				}
				if (isSatisfied[ic])
					continue;
				tmpSol = solution;

				//Fourth, loop over all parameter by forcing them (not random)
				for (size_t j = 0; j < nonZeroIndices.n_elem; j++)
				{
					index = nonZeroIndices(j);
					//change the parameter
					int n_sec =16;
					arma::uvec sec = arma::randperm(n_sec + 1);//random
					//n_sec =9;
					//arma::uvec secNumb = { 0,1,2,4,8,12,14,15,16 };//
					//arma::uvec sec = secNumb(arma::randperm(secNumb.n_elem));
					for (size_t k = 0; k < sec.n_elem; k++)
					{
						ElemType Fraction = static_cast<ElemType>(sec[k]) / n_sec;
						tmpSol[index] = Fraction * (upperbounds(index) - lowerbounds(index)) + lowerbounds(index);
						if (checkInequalityConstraint(tmpSol, ic))
						{
							solution = tmpSol;
							isSatisfied[ic] = true;
							break;
						}
					}
					if (isSatisfied[ic])
						break;
				}


			}
			isSatisfied = checkInequalityConstraints(solution);
			if (!std::all_of(isSatisfied.begin(), isSatisfied.end(), [](bool valid) { return valid; }))
			{
				#pragma message("Warning: Invalid validity found")
				for (size_t l = 0; l < isSatisfied.size(); l++)
				{
					if (!isSatisfied[l])
					{
						std::cout << "inequality constraint number:" << l << " is still incompatible"<< std::endl;
					}
				}
			}


		}


		void applyExpressions(BaseMatType& solution, std::vector<bool>& isSatisfied)
		{
			//std::vector<bool> isSatisfied = checkInequalityConstraints(solution);
			for (size_t ic = 0; ic < isSatisfied.size(); ic++) //loop over each inequality constrains 
			{
				if (isSatisfied[ic] == true)
					continue;
				if (checkExpressionConstraint(solution, ic))
					continue;
				BaseMatType  tmpSol = solution;

				arma::uvec nonZeroIndices = arma::find(constrainedIdxExpr.row(ic));

				//first try to change the first parameter that is included in the constrain
				//the first parameter in the constraint is considered the most important one
				int index = nonZeroIndices(0);
				arma::vec rnd = arma::randu(20);
				arma::uvec randIndext = arma::randperm(20);
				for (size_t j = 0; j < 20; j++)
				{
					tmpSol[index] = rnd[randIndext(j)] * (upperbounds(index) - lowerbounds(index)) + lowerbounds(index);
					if (checkExpressionConstraint(tmpSol, ic))
					{
						solution = tmpSol;
						isSatisfied[ic] = true;
						break;
					}
				}

				if (isSatisfied[ic])
					continue;
				tmpSol = solution;

				//Second try to change the other parameters that are included ONLY in this constrain
				arma::uvec parameterToChange = nonZeroIndices(arma::randperm(nonZeroIndices.n_elem));
				arma::uvec action = arma::randperm(3);
				std::vector<int> repeatedParameter;
				for (size_t j = 0; j < parameterToChange.n_elem; j++) //loop over parameter to be change 
				{
					index = parameterToChange(j);
					if (index == nonZeroIndices(0))
						continue;
					//check if the parameter is included in other constrains
					arma::uvec tmp = arma::find(constrainedIdxExpr.col(index));
					if (tmp.n_elem > 1)
					{
						repeatedParameter.push_back(index);
						continue;
					}
					//change the parameter
					arma::vec rnd = arma::randu(20);
					for (size_t j = 0; j < rnd.n_elem; j++)
					{
						tmpSol[index] = rnd[j] * (upperbounds(index) - lowerbounds(index)) + lowerbounds(index);
						if (checkExpressionConstraint(tmpSol, ic))
						{
							solution = tmpSol;
							isSatisfied[ic] = true;
							break;
						}
					}
				}

				if (isSatisfied[ic])
					continue;
				tmpSol = solution;

				//third, change the other parameters that are included in several constrains 
				for (size_t j = 0; j < repeatedParameter.size(); j++) //loop over parameter that are included in several constrains 
				{
					index = repeatedParameter[j];
					if (index == nonZeroIndices(0))
						continue;
					//change the parameter
					arma::vec rnd = arma::randu(20);
					for (size_t j = 0; j < rnd.n_elem; j++)
					{
						tmpSol[index] = rnd[j] * (upperbounds(index) - lowerbounds(index)) + lowerbounds(index);
						if (checkExpressionConstraint(tmpSol, ic))
						{
							solution = tmpSol;
							isSatisfied[ic] = true;
							break;
						}
					}
					if (isSatisfied[ic])
						break;
				}
				if (isSatisfied[ic])
					continue;
				tmpSol = solution;

				//Fourth, loop over all parameter by forcing them (not random)
				for (size_t j = 0; j < nonZeroIndices.n_elem; j++)
				{
					index = nonZeroIndices(j);
					//change the parameter
					int n_sec = 16;
					arma::uvec sec = arma::randperm(n_sec + 1);//random
					//n_sec =9;
					//arma::uvec secNumb = { 0,1,2,4,8,12,14,15,16 };//
					//arma::uvec sec = secNumb(arma::randperm(secNumb.n_elem));
					for (size_t k = 0; k < sec.n_elem; k++)
					{
						ElemType Fraction = static_cast<ElemType>(sec[k]) / n_sec;
						tmpSol[index] = Fraction * (upperbounds(index) - lowerbounds(index)) + lowerbounds(index);
						if (checkExpressionConstraint(tmpSol, ic))
						{
							solution = tmpSol;
							isSatisfied[ic] = true;
							break;
						}
					}
					if (isSatisfied[ic])
						break;
				}


			}
			isSatisfied = checkExpressionsConstraints(solution);
			if (!std::all_of(isSatisfied.begin(), isSatisfied.end(), [](bool valid) { return valid; }))
			{
				#pragma message("Warning: Invalid validity found")
				for (size_t l = 0; l < isSatisfied.size(); l++)
				{
					if (!isSatisfied[l])
					{
						//std::cout << "Expression constraint number:" << l << " is still incompatible" << std::endl;
					}
				}
			}


		}



		void SetBounds(arma::Col<ElemType>& lowerbound, arma::Col<ElemType>& upperbound) {
			lowerbounds = lowerbound;  upperbounds = upperbound;
		}


		
		arma::Col<ElemType> get_lower_bound() const
		{
			if (_initial_values==nullptr)
				return lowerbounds;
			else
				return lowerbounds - arma::conv_to<arma::Col<ElemType>>::from(*_initial_values);
			
		}
		arma::Col<ElemType> get_upper_bound() const
		{
			if (_initial_values == nullptr)
				return upperbounds;
			else
				return upperbounds - arma::conv_to<arma::Col<ElemType>>::from(*_initial_values);
		}

		void setInequalityConstraints(const arma::imat& constrainedIdxInq, const arma::Col<ElemType>& threshold, std::vector<InequalityType> inqtyp, const ConstraintType& type=ens::ConstraintType::inequality)
		{
			if (std::find(typOfConstraints.begin(), typOfConstraints.end(), type) != typOfConstraints.end())
			{
				throw std::runtime_error("The constraint type is already set");
				return;
			}
			//this->type = type;
			typOfConstraints.push_back(type);
			this->threshold = threshold;
			this->inqualityType = inqtyp;
			this->constrainedIdxInq = constrainedIdxInq;
			constraintCounts[ConstraintType::inequality] = constrainedIdxInq.n_rows;
		}

		void setExpressionConstraints(const std::vector<std::string>& strExpressions, const ConstraintType& type = ens::ConstraintType::expression)
		{
			if (std::find(typOfConstraints.begin(), typOfConstraints.end(), type) != typOfConstraints.end())
			{
				throw std::runtime_error("The constraint type is already set");
				return;
			}
			typOfConstraints.push_back(type);
			this->strExpressions = strExpressions;
			dimProb=lowerbounds.n_rows;
			//compiling the expressions
			//initializing with a dummy vector
			prmtrVector = std::vector<ElemType>(lowerbounds.begin(), lowerbounds.end());
			symbolTable.add_vector("x", prmtrVector); // Dummy vector; will be updated during evaluation.
			symbolTable.add_constants();

			// Compile.
			//for (const std::string& exprStr : strExpressions) {
			for (std::string exprStr : strExpressions) {
				//std::cout << "Expression:" << exprStr << std::endl;
				exprtk::expression<ElemType> compiledExpression;
				compiledExpression.register_symbol_table(symbolTable);

				if (!exprtk::parser<ElemType>().compile(exprStr, compiledExpression)) {
					throw std::runtime_error("Error compiling constraint expression. Please Check the constraint expression"+ exprStr);
				}
				constraintExpressions.push_back(compiledExpression);


				// Find and store the parameter indices in the expression
				std::vector<int> indices;
				arma::irowvec indices1(dimProb, arma::fill::zeros);
				std::size_t found = exprStr.find("x[");
				while (found != std::string::npos) {
					std::size_t end = exprStr.find("]", found);
					if (end != std::string::npos) {
						std::string indexStr = exprStr.substr(found + 2, end - found - 2);
						int index = std::stoi(indexStr);
						if (index >= lowerbounds.n_elem) {
							throw std::runtime_error("Index in expression exceeds the size of parameter.");
						}
						if (std::find(indices.begin(), indices.end(), index) == indices.end())
						indices.push_back(index);
						indices1(index) = 1;
						found = exprStr.find("x[", end);
					}
				}
				exprPrmtrIndcs.push_back(indices);
				constrainedIdxExpr.insert_rows(constrainedIdxExpr.n_rows, indices1);
			}
		}



		void printConstraints() const
		{
			std::cout << "The constraints are: " << std::endl;
			for (size_t i = 0; i < typOfConstraints.size(); i++)
			{
				switch (typOfConstraints[i])
				{
					case ConstraintType::bounds:
						std::cout << i<<". Bounds Constraint:" << std::endl;
						std::cout << "Lower bound < " << "Parameter " << "< Upper bound" << std::endl;
						for (size_t j = 0; j < lowerbounds.n_elem; j++)
						{
							std::cout << lowerbounds(j) << " < " << "Parameter [" << j << "]  < " << upperbounds(j) << std::endl;
						}
						break;
					case ConstraintType::inequality:
						{
							std::cout << std::endl << i << ". Inequality Constraints:" << std::endl;
							std::cout << "Parameter vector " << "Inequality type " << "Threshold " << std::endl;
							for (size_t j = 0; j < constrainedIdxInq.n_rows; j++)
							{
								std::cout << "inequality constraint" << j << ":" << std::endl;
								int count=0;
								for (size_t k = 0; k < constrainedIdxInq.row(j).n_cols; k++)
								{
									if (constrainedIdxInq(j, k) == 0)
										continue;
									if (count)
										std::cout << " + ";
									std::cout << constrainedIdxInq(j, k) << "[x" << k << "]";
									count= 1;
									//if (k < constrainedIdxInq.row(j).n_cols - 1)
									//	std::cout << " + ";
								}
								switch (inqualityType[j])
								{
								case InequalityType::less_than:
									std::cout << " < " << threshold(j) << std::endl;
									break;
								case InequalityType::less_than_or_equal_to:
									std::cout << " <= " << threshold(j) << std::endl;
									break;
								case InequalityType::greater_than:
									std::cout << " > " << threshold(j) << std::endl;
									break;
								case InequalityType::greater_than_or_equal_to:
									std::cout << " >= " << threshold(j) << std::endl;
									break;
								case InequalityType::equal_to:
									std::cout << " = " << threshold(j) << std::endl;
									break;
								default:
									throw std::runtime_error("Invalid inequality type.");
								}
							}
							break;
						}
					case ConstraintType::equality:
						std::cout << std::endl << i << ". Equality" << std::endl;
						break;
					case ConstraintType::expression:
						std::cout << std::endl << i << ". Expression Constraints:" << std::endl;
							for (size_t j = 0; j < constraintExpressions.size(); j++)
							{
								std::cout << "Expression " << j << ": " << strExpressions[j] << std::endl;
								std::cout << "Involved Parameter Indices: ";
								for (int index : exprPrmtrIndcs[j])
								{
									std::cout << index << " ";
								}
								std::cout << std::endl;
							}
						break;
					default:
						throw std::runtime_error("Invalid constraint type.");
				}
			}
		}

		void printValidity(BaseMatType& solution) 
		{
			for (size_t i = 0; i < solution.size(); i++)
			{
				std::cout << "Parameter x[" << i << "] = " << solution(i) << std::endl;
			}
			std::cout << "Non validated constraints are: " << std::endl;
			for (size_t i = 0; i < typOfConstraints.size(); i++)
			{
				switch (typOfConstraints[i])
				{
				case ConstraintType::bounds:
					{
						std::vector<bool>  validBound = checkValidity(solution, ens::ConstraintType::bounds);
						std::cout << i << ". Bounds Constraint:" << std::endl;
						for (size_t j = 0; j < lowerbounds.n_elem; j++)
						{
							if (!validBound[j])
								std::cout << lowerbounds(j) << " < "  << solution(j) << "  < " << upperbounds(j) << "  Parameter [" << j << "] " << "is not valid" << std::endl;
						}
						break;
					}
				case ConstraintType::inequality:
					{
						std::cout << std::endl << i << ". Inequality Constraints:" << std::endl;
						std::vector<bool>  validIneq = checkValidity(solution, ens::ConstraintType::inequality);
						for (size_t j = 0; j < constrainedIdxInq.n_rows; j++)
						{
							if (validIneq[j])
								continue;
							std::cout << "inequality constraint " << j << ": is not valid" << std::endl;
							bool count = 0;
							for (size_t k = 0; k < constrainedIdxInq.row(j).n_cols; k++)
							{
								if (constrainedIdxInq(j, k) == 0)
									continue;
								if (count)
									std::cout << " + ";
								std::cout << constrainedIdxInq(j, k) << "[" << solution(k) << "]";
								count = 1;
								//if (k < constrainedIdxInq.row(j).n_cols - 1)
								//	std::cout << " + ";
							}
							switch (inqualityType[j])
							{
							case InequalityType::less_than:
								std::cout << " < " << threshold(j) << std::endl;
								break;
							case InequalityType::less_than_or_equal_to:
								std::cout << " <= " << threshold(j) << std::endl;
								break;
							case InequalityType::greater_than:
								std::cout << " > " << threshold(j) << std::endl;
								break;
							case InequalityType::greater_than_or_equal_to:
								std::cout << " >= " << threshold(j) << std::endl;
								break;
							case InequalityType::equal_to:
								std::cout << " = " << threshold(j) << std::endl;
								break;
							default:
								throw std::runtime_error("Invalid inequality type.");
							}
						}
						break;
					}
					
				case ConstraintType::equality:
					std::cout << std::endl << i << ". Equality" << std::endl;
					break;
				case ConstraintType::expression:
					{
						std::cout << std::endl << i << ". Expression Constraints:" << std::endl;
						std::vector<bool>  validExpr = checkValidity(solution, ens::ConstraintType::expression);
						for (size_t j = 0; j < constraintExpressions.size(); j++)
						{
							if (validExpr[j])
								continue;
							std::cout << "Expression " << j << ": " << strExpressions[j] << " is not valid " << std::endl;
							std::cout << "Involved Parameter Indices: ";
							for (int index : exprPrmtrIndcs[j])
							{
								std::cout << "x[" << index << "]= " << solution(index) << " , ";
							}
							std::cout << std::endl;
						}
						break;
					}
				default:
					throw std::runtime_error("Invalid constraint type.");
				}
			}
		}

		void applyConstraints(arma::Cube<ElemType>& particlePositions) 
		{
			switch (type)
			{
			case ConstraintType::bounds:
				applyBoundsConstraint(particlePositions);
				break;
			case ConstraintType::inequality:
				applyInequalityConstraints(particlePositions);
				break;
			case ConstraintType::equality:
				//applyEqualityConstraints(particlePositions, threshold, constrainedIdxInq);
				break;
			case ConstraintType::expression:
				applyExpressionsConstraints(particlePositions);
				break;
			default:
				throw std::runtime_error("Invalid constraint type.");
			}

		}


		void applyInequalityConstraints(arma::Cube<ElemType>& particlePositions) const
		{
			std::vector<std::vector<bool>> validity = validateInequality(particlePositions);
			for (size_t i = 0; i < particlePositions.n_slices; i++)
			{
				if (std::all_of(validity[i].begin(), validity[i].end(), [](bool value) { return value; })) {
					continue;
				}
				std::vector<bool> particleValidation = validity[i];
				arma::Cube<ElemType> copyParticlePositions = particlePositions;
				/*std::vector<bool> particleValidation= validateExpressionsParticle(particlePositions.slice(i));*/

				for (size_t jj = 0; jj < copyParticlePositions.n_rows; jj++)//loop over parameters
				{
					if (particleValidation[jj] == true)
						continue;
					arma::Cube<ElemType> copyCopyParticlePositions = copyParticlePositions;
					for (size_t ll = 0; ll < 20; ll++) {
						randomValue(copyCopyParticlePositions(jj, 0, i), jj);
						particleValidation = validateInequalityParticle(copyCopyParticlePositions.slice(i));
						if (particleValidation[jj] == true) {
							copyParticlePositions = copyCopyParticlePositions;
							break;
						}
					}
				}
				// check just one slice
				particleValidation = validateInequalityParticle(copyParticlePositions.slice(i));
				if (std::all_of(particleValidation.begin(), particleValidation.end(), [](bool value) { return value; })) {
					particlePositions = copyParticlePositions;
					break;
				}
				copyParticlePositions = particlePositions;
				particleValidation = validateInequalityParticle(copyParticlePositions.slice(i));
				for (size_t j = 0; j < particlePositions.n_rows; j++)
				{
					if (particleValidation[j] == true)
						continue;
					arma::Cube<ElemType> tmpPrtclPstns = copyParticlePositions;
					//RandBehave(particlePositions.slice(i));
					tmpPrtclPstns(j, 0, i) = upperbounds(j);
					particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
					if (particleValidation[j] == true) {
						copyParticlePositions = tmpPrtclPstns;
						continue;
					}
						
				
					tmpPrtclPstns(j, 0, i) = lowerbounds(j);
					particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
					if (particleValidation[j] == true) {
						copyParticlePositions = tmpPrtclPstns;
						continue;
					}
					tmpPrtclPstns = copyParticlePositions;
					//change other parameter
					for (size_t k = 0; k < constrainedIdxInq.n_rows; k++)
					{
						if (constrainedIdxInq(j, k) == 0)
							continue;
						if (k == j)
							continue;
						arma::Cube<ElemType> tmpPrtclPstns = copyParticlePositions;
						tmpPrtclPstns(k, 0, i) = (upperbounds(k) + lowerbounds(k)) / 2;
						particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
						if (particleValidation[j] == true) {
							copyParticlePositions = tmpPrtclPstns;
							break;
						}
						tmpPrtclPstns(k, 0, i) = (upperbounds(k) + lowerbounds(k)) / 4;
						particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
						if (particleValidation[j] == true) {
							copyParticlePositions = tmpPrtclPstns;
							break;
						}
						tmpPrtclPstns(k, 0, i) = (upperbounds(k) + lowerbounds(k)) * 3 / 4;
						particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
						if (particleValidation[j] == true) {
							copyParticlePositions = tmpPrtclPstns;
							break;
						}
						tmpPrtclPstns(k, 0, i) = (upperbounds(k) + lowerbounds(k)) / 8;
						particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
						if (particleValidation[j] == true) {
							copyParticlePositions = tmpPrtclPstns;
							break;
						}
						tmpPrtclPstns(k, 0, i) = (upperbounds(k) + lowerbounds(k)) * 7 / 8;
						particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
						if (particleValidation[j] == true) {
							copyParticlePositions = tmpPrtclPstns;
							break;
						}
						tmpPrtclPstns(k, 0, i) = (upperbounds(k) + lowerbounds(k)) / 16;
						particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
						if (particleValidation[j] == true) {
							copyParticlePositions = tmpPrtclPstns;
							break;
						}
						tmpPrtclPstns(k, 0, i) = (upperbounds(k) + lowerbounds(k)) * 15 / 16;
						particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
						if (particleValidation[j] == true) {
							copyParticlePositions = tmpPrtclPstns;
							break;
						}
						tmpPrtclPstns(k, 0, i) = upperbounds(k);
						particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
						if (particleValidation[j] == true) {
							copyParticlePositions = tmpPrtclPstns;
							break;
						}
						tmpPrtclPstns(k, 0, i) = lowerbounds(k);
						particleValidation = validateInequalityParticle(tmpPrtclPstns.slice(i));
						if (particleValidation[j] == true) {
							copyParticlePositions = tmpPrtclPstns;
							break;
						}
					}
				}
				//if (!std::all_of(particleValidation.begin(), particleValidation.end(), [](bool valid)
				//{
				//	return valid;
				//}))
				//{
				//	throw std::runtime_error("Invalid validity found");
				//}



				//if (!std::all_of(constraintSatisfactions.begin(), constraintSatisfactions.end(), [](bool valid) { return valid; }))
				particlePositions = copyParticlePositions;
				validity = validateInequality(particlePositions);
				if (!std::all_of(validity[i].begin(), validity[i].end(), [](bool valid) {
					return valid;
					}))
				{
					throw std::runtime_error("Invalid validity found");//maybe Over-constrained
				}
			}
			validity = validateInequality(particlePositions);

			bool allValid = true;

			for (const auto& innerVec : validity) {
				if (!std::all_of(innerVec.begin(), innerVec.end(), [](bool valid) {
					return valid;
					})) {
					allValid = false;
					break;  // No need to continue checking if any inner vector contains false.
				}
			}

			if (!allValid) {
				//throw std::runtime_error("Invalid validity found");
				std::cout << "Invalid validity found on particle " << std::endl;
			}

		}


		std::vector<std::vector<bool>> validateInequality(arma::Cube<ElemType>& particlePositions) const {

			// Ensure 
			if (threshold.n_elem != constrainedIdxInq.n_rows)
			{
				throw std::runtime_error("Threshold and constrainedIdxInq dimensions do not match.");
			}
			// Ensure constrainedIdxInq is symmetric
			if (arma::any(arma::vectorise(constrainedIdxInq) != arma::vectorise(constrainedIdxInq.t())))
			{
				throw std::runtime_error("constrainedIdxInq is not symmetric.");
			}

			std::vector<std::vector<bool>> constraintSatisfactions(particlePositions.n_slices, std::vector<bool>(particlePositions.n_rows, false));

			for (size_t i = 0; i < particlePositions.n_slices; i++) {


				
				double sum1 = 0.0;
				arma::vec particle = arma::vectorise(particlePositions.slice(i));

				for (size_t j = 0; j < particle.n_elem; j++) {

					if (arma::accu(constrainedIdxInq.row(j)) == 0 || inqualityType[j] == noConstrains)
					{
						continue; // No constraints for this entry.
					}
					for (size_t k = 0; k < constrainedIdxInq.n_cols; k++) {
						int index1 = constrainedIdxInq(j, k);
						if (index1 == 0 || std::abs(index1) >= static_cast<int>(particlePositions.n_cols))
						{
							continue;
						}
						sum1 += particle(j) * index1;
					}

					switch (inqualityType[j])
					{
					case InequalityType::less_than:
						if (sum1 < threshold(j))
						{
							constraintSatisfactions[i][j] = true; // Constraint violated for this index.
						}
						break;
					case InequalityType::less_than_or_equal_to:
						if (sum1 <= threshold(i))
						{
							constraintSatisfactions[i][j] = true; // Constraint violated for this index.
						}
						break;
					case InequalityType::greater_than:
						if (sum1 > threshold(i))
						{
							constraintSatisfactions[i][j] = true; // Constraint violated for this index.
						}
						break;
					case InequalityType::greater_than_or_equal_to:

						if (sum1 >= threshold(j))
						{
							constraintSatisfactions[i][j] = true; // Constraint violated for this index.
						}
						break;
					case InequalityType::equal_to:
						if (sum1 == threshold(j))
						{
							constraintSatisfactions[i][j] = true; // Constraint violated for this index.
						}
						break;
					default:
						throw std::runtime_error("Invalid inequality type.");
					}

					
				}

				return constraintSatisfactions;

			}

			
		}

		std::vector<bool> validateInequalityParticle(BaseMatType& particle) const {
			arma::Cube<ElemType> tempCube(particle.n_rows, particle.n_cols, 1);
			tempCube.slice(0) = particle;
			std::vector<std::vector<bool>> particleConstraintSatisfactions = validateInequality(tempCube);
			std::vector<bool> particleValidation = particleConstraintSatisfactions[0];
			return particleValidation;
		}

		std::vector<std::vector<bool>> validateExpressions(arma::Cube<ElemType>& particlePositions)
		{
			std::vector<std::vector<bool>> constraintSatisfactions(particlePositions.n_slices,
				std::vector<bool>(constraintExpressions.size(), false));

			for (size_t i = 0; i < particlePositions.n_slices; i++)//loop over particles
			{
				arma::Col<ElemType> particle = arma::vectorise(particlePositions.slice(i));
				std::vector<ElemType> prmtrVector1 = arma::conv_to<std::vector<ElemType>>::from(particle);
				prmtrVector = prmtrVector1;
				std::cout << "Particle: " << std::endl << particle << std::endl;
				// Evaluate the expressions
				for (size_t k = 0; k < constraintExpressions.size(); k++)//loop over Expressions consentrains  
				{
					std::cout << "Expression " << k << ": " << strExpressions[k] << std::endl;
					std::cout << "Parameter Indices: ";
					for (int index : exprPrmtrIndcs[k])
					{
						std::cout << index << " ";
					}
					std::cout << std::endl;
					std::cout << "Result: " << constraintExpressions[k].value() << std::endl;
					constraintSatisfactions[i][k] = constraintExpressions[k].value();
				}
			}
			return constraintSatisfactions;
		}

		std::vector<bool> validateExpressionsParticle(BaseMatType& particle)  {
			arma::Cube<ElemType> tempCube(particle.n_rows, particle.n_cols, 1);
			tempCube.slice(0) = particle;
			std::vector<std::vector<bool>> particleConstraintSatisfactions = validateExpressions(tempCube);
			std::vector<bool> particleValidation = particleConstraintSatisfactions[0];
			return particleValidation;
		}

		void applyExpressionsConstraints(arma::Cube<ElemType>& particlePositions) 
		{
			std::vector<std::vector<bool>> validity = validateExpressions(particlePositions);
			for (size_t i = 0; i < particlePositions.n_slices; i++)
			{
				if (std::all_of(validity[i].begin(), validity[i].end(), [](bool value) { return value; })) {
					continue;
				}
				std::vector<bool> particleValidation = validity[i];
				arma::Cube<ElemType> copyParticlePositions = particlePositions;
				/*std::vector<bool> particleValidation= validateExpressionsParticle(particlePositions.slice(i));*/
				for (size_t jj = 0; jj < exprPrmtrIndcs.size(); jj++)//loop over expressions
				{
					if (particleValidation[jj] == true)
						continue;
					arma::Cube<ElemType> copyCopyParticlePositions = copyParticlePositions;
					for (size_t kk = 0; kk < exprPrmtrIndcs[jj].size(); kk++) //loop over parameters involved in the expression
					{
						size_t pIndex = exprPrmtrIndcs[jj][kk];
						for (size_t ll = 0; ll < 20; ll++) {
							randomValue(copyCopyParticlePositions(pIndex, 0, i), pIndex);
							particleValidation = validateExpressionsParticle(copyCopyParticlePositions.slice(i));
							if (particleValidation[jj] == true) {
								copyParticlePositions = copyCopyParticlePositions;
								break;
							}
						}
						if (particleValidation[jj] == true) {
							break;
						}
					}
						
				}
					// check just one particle, if all expression are validate, keep new data, otherwise remove it
					particleValidation = validateExpressionsParticle(copyParticlePositions.slice(i));
				if (std::all_of(particleValidation.begin(), particleValidation.end(), [](bool value) { return value; })) {
					particlePositions = copyParticlePositions;
					break;
				}
				copyParticlePositions = particlePositions;


				particleValidation = validateExpressionsParticle(copyParticlePositions.slice(i));


				for (size_t j = 0; j < particlePositions.n_rows; j++)// loop over parameters
				{
					std::vector<int> exprWthThsPrmeter;// parameters that are related to each expression
					for (size_t jj = 0; jj < exprPrmtrIndcs.size(); jj++)//loop over expressions
					{
						for (size_t kk = 0; kk < exprPrmtrIndcs[jj].size(); kk++) //loop over parameters involved in the expression
						{
							if (exprPrmtrIndcs[jj][kk] == j)
								exprWthThsPrmeter.push_back(jj);
						}
					}

					arma::Cube<ElemType> tmpPrtclPstns = copyParticlePositions;
					particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
					bool allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
						[&particleValidation](int index) {
							return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
						});
					if (allTrue) {
						copyParticlePositions = tmpPrtclPstns;
						continue;
					}


						tmpPrtclPstns(j, 0, i) = (upperbounds(j) + lowerbounds(j)) / 2;
						particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
						allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
							[&particleValidation](int index) {
								return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
							});
						if (allTrue) {
							copyParticlePositions = tmpPrtclPstns;
							continue;
						}

						tmpPrtclPstns(j, 0, i) = (upperbounds(j) + lowerbounds(j)) / 4;
						particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
						allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
							[&particleValidation](int index) {
								return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
							});
						if (allTrue) {
							copyParticlePositions = tmpPrtclPstns;
							continue;
						}
						tmpPrtclPstns(j, 0, i) = (upperbounds(j) + lowerbounds(j)) * 3 / 4;
						particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
						allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
							[&particleValidation](int index) {
								return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
							});
						if (allTrue) {
							copyParticlePositions = tmpPrtclPstns;
							continue;
						}
						tmpPrtclPstns(j, 0, i) = (upperbounds(j) + lowerbounds(j)) / 8;
						particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
						allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
							[&particleValidation](int index) {
								return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
							});
						if (allTrue) {
							copyParticlePositions = tmpPrtclPstns;
							continue;
						}
						tmpPrtclPstns(j, 0, i) = (upperbounds(j) + lowerbounds(j)) * 7 / 8;
						particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
						allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
							[&particleValidation](int index) {
								return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
							});
						if (allTrue) {
							copyParticlePositions = tmpPrtclPstns;
							continue;
						}
						tmpPrtclPstns(j, 0, i) = (upperbounds(j) + lowerbounds(j)) / 16;
						particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
						allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
							[&particleValidation](int index) {
								return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
							});
						if (allTrue) {
							copyParticlePositions = tmpPrtclPstns;
							continue;
						}
						tmpPrtclPstns(j, 0, i) = (upperbounds(j) + lowerbounds(j)) * 15 / 16;
						particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
						allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
							[&particleValidation](int index) {
								return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
							});
						if (allTrue) {
							copyParticlePositions = tmpPrtclPstns;
							continue;
						}
						tmpPrtclPstns(j, 0, i) = upperbounds(j);
						particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
						allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
							[&particleValidation](int index) {
								return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
							});
						if (allTrue) {
							copyParticlePositions = tmpPrtclPstns;
							continue;
						}
						tmpPrtclPstns(j, 0, i) = lowerbounds(j);
						particleValidation = validateExpressionsParticle(tmpPrtclPstns.slice(i));
						allTrue = std::all_of(exprWthThsPrmeter.begin(), exprWthThsPrmeter.end(),
							[&particleValidation](int index) {
								return index >= 0 && static_cast<size_t>(index) < particleValidation.size() && particleValidation[static_cast<size_t>(index)];
							});
						if (allTrue) {
							copyParticlePositions = tmpPrtclPstns;
							continue;
						}
					
				}
				//if (!std::all_of(particleValidation.begin(), particleValidation.end(), [](bool valid)
				//{
				//	return valid;
				//}))
				//{
				//	throw std::runtime_error("Invalid validity found");
				//}



				//if (!std::all_of(constraintSatisfactions.begin(), constraintSatisfactions.end(), [](bool valid) { return valid; }))
				particlePositions = copyParticlePositions;
				validity = validateExpressions(particlePositions);
				if (!std::all_of(validity[i].begin(), validity[i].end(), [](bool valid) {
					return valid;
					}))
				{
					throw std::runtime_error("Invalid validity found");
				}
			}
			validity = validateExpressions(particlePositions);

			bool allValid = true;

			for (const auto& innerVec : validity) {
				if (!std::all_of(innerVec.begin(), innerVec.end(), [](bool valid) {
					return valid;
					})) {
					allValid = false;
					break;  // No need to continue checking if any inner vector contains false.
				}
			}

			if (!allValid) {
				//throw std::runtime_error("Invalid validity found");
				std::cout << "Invalid validity found on particle " << std::endl;
			}

		}

		void applyBoundsConstraint(arma::Cube<ElemType>& particlePositions) const
		{
			std::vector<bool> validity = validateBounds(particlePositions);
			for (size_t j = 0; j < particlePositions.n_slices; j++)
			{
				if (validity[j] == true)
					continue;
				RandBehave(particlePositions.slice(j));

			}
			
			validity = validateBounds(particlePositions);

			if (!std::all_of(validity.begin(), validity.end(), [](bool valid) { return valid; }))
			{
				throw std::runtime_error("Invalid validity found");
			}
		}

		std::vector<bool> validateBounds(const arma::Cube<ElemType>& particlePositions) const
		{
			std::vector<bool> constraintSatisfactions(particlePositions.n_slices, true);

			for (size_t i = 0; i < particlePositions.n_slices; i++) {
				arma::vec particle = arma::vectorise(particlePositions.slice(i));

				for (size_t j = 0; j < particle.n_elem; j++) {
					if (particle(j) < lowerbounds(j) || particle(j) > upperbounds(j)) {
						constraintSatisfactions[i] = false;
						break;
					}
				}
			}

			return constraintSatisfactions;
		}

	

		inline BaseMatType RandomDisplacement(const BaseMatType& seed)
		{
			BaseMatType population(seed.n_rows, seed.n_cols);
			population = seed + arma::randn(seed.n_rows, seed.n_cols);
			RandomReseed(population);
			return population;
		}

		inline void RandBehave(BaseMatType& population) const
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

		inline void Stick2box(BaseMatType& population) const
		{
			if (lowerbounds.is_empty()) return;

			for (size_t ii = 0; ii < population.n_rows; ii++)
			{
				population.row(ii).clamp(lowerbounds(ii), upperbounds(ii));
			}
		}

		inline void Bounce(BaseMatType& population) const
		{
			if (lowerbounds.is_empty()) return;
			for (size_t ii = 0; ii < population.n_rows; ii++)
			{
				for (size_t jj = 0; jj < population.n_cols; jj++)
				{
					if (population(ii, jj) < lowerbounds(ii))
					{
						ElemType infeasible_length = (lowerbounds(ii) - population(ii, jj)) / (upperbounds(ii) - lowerbounds(ii));
						population(ii, jj) = lowerbounds(ii) + std::min(infeasible_length, (ElemType)1) * (upperbounds(ii) - lowerbounds(ii));
					}
					if (population(ii, jj) > upperbounds(ii))
					{
						ElemType infeasible_length = (population(ii, jj) - upperbounds(ii)) / (upperbounds(ii) - lowerbounds(ii));
						population(ii, jj) = lowerbounds(ii) + (1 - std::min(infeasible_length, (ElemType)1)) * (upperbounds(ii) - lowerbounds(ii));
					}
				}
			}
		}

		inline void RandomReseed(BaseMatType& population) const
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

		inline void randomValue(ElemType& value,const int& indx) const
		{
			auto rnd = arma::randu();
			value = lowerbounds[indx] + (ElemType)rnd * (upperbounds[indx] - lowerbounds[indx]);
		}

		inline void bounceValue(ElemType& value, const int& indx) const
		{
			if (value < lowerbounds[indx])
			{
				ElemType infeasible_length = (lowerbounds[indx] - value) / (upperbounds[indx] - lowerbounds[indx]);
				value = lowerbounds[indx] + std::min(infeasible_length, (ElemType)1) * (upperbounds[indx] - lowerbounds[indx]);
			}
			if (value > upperbounds[indx])
			{
				ElemType infeasible_length = (value - upperbounds[indx]) / (upperbounds[indx] - lowerbounds[indx]);
				value = lowerbounds[indx] + (1 - std::min(infeasible_length, (ElemType)1)) * (upperbounds[indx] - lowerbounds[indx]);
			}
		}

		inline void sequentialValue(ElemType& value, const int& indx) const
		{
			arma::uvec sec = arma::randperm(10);
			value = lowerbounds[indx] + (sec/8) * (upperbounds[indx] - lowerbounds[indx]);
		}

		inline void getValue(ElemType& value, const int& indx, const double& fraction  ) const
		{

			value = lowerbounds[indx] + static_cast<ElemType>(fraction) * (upperbounds[indx] - lowerbounds[indx]);
		}



	};
}

#endif