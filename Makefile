todo: cuda_test1 cuda_test2 cuda_final_basic cuda_test3 cuda_test4 cuda_test5 cuda_test6 cuda_final_shared_reduction cuda_final_shared_memory cuda_final_shared_reduction_improved
cuda_test1: cuda_test1.cu 
	nvcc -o cuda_test1 cuda_test1.cu  -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_test2: cuda_test2.cu
	nvcc -o cuda_test2 cuda_test2.cu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_final_basic: cuda_final_basic.cu
	nvcc -o cuda_final_basic cuda_final_basic.cu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_test3: cuda_test3.cu
	nvcc -o cuda_test3 cuda_test3.cu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_test4: cuda_test4.cu
	nvcc -o cuda_test4 cuda_test4.cu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_test5: cuda_test5.cu
	nvcc -o cuda_test5 cuda_test5.cu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_test6: cuda_test6.cu
	nvcc -o cuda_test6 cuda_test6.cu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_final_shared_reduction: cuda_final_shared_reduction.cu
	nvcc -o cuda_final_shared_reduction cuda_final_shared_reduction.cu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_final_shared_memory: cuda_final_shared_memory.cu
	nvcc -o cuda_final_shared_memory cuda_final_shared_memory.cu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_final_shared_reduction_improved: cuda_final_shared_reduction_improved.cu
	nvcc -o cuda_final_shared_reduction_improved cuda_final_shared_reduction_improved.cu -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
clean:
	rm cuda_test1 cuda_test2 cuda_final_basic cuda_test3 cuda_test4 cuda_test5 cuda_test6 cuda_final_shared_reduction cuda_final_shared_memory cuda_final_shared_reduction_improved



