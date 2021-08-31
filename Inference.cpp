#include "inference.h"



std::string OcrClass::inference(const char* image_path) {

	auto t_start_1 = std::chrono::high_resolution_clock::now();
	//读图加预处理
	Mat img_float_1, img_pad_1, img_resized, proba_map_erode_1, proba_map_dilate_1, img;
	Mat image = imread(image_path);
	resize(image, img, Size(2048, 1536));
	double resize_scale = double(image_size_db) / double(img.size().width);
	resize(img, img_resized, Size(), resize_scale, resize_scale);
	//resize(img, img_resized, img_size);
	//Size img_size = Size(320, 240);
	int pad_size = image_size_db - img.size().height * resize_scale;
	copyMakeBorder(img_resized, img_pad_1, 0, pad_size, 0, 0, cv::BORDER_CONSTANT, Scalar(0));
	img_pad_1.convertTo(img_float_1, CV_32FC3, 1 / 255.0, 0.0);		//归一化
	float* image_ptr_1 = (float*)img_float_1.data;

	// 创建输入tensor
	Ort::Value input_tensor_1 = Ort::Value::CreateTensor<float>(memory_info, image_ptr_1, input_tensor_size, input_node_dims.data(), 4);
	// 运行
	auto output_tensors_1 = session_1.Run(Ort::RunOptions{ nullptr }, input_node_name_1.data(), &input_tensor_1, 1, output_node_name_1.data(), 1);
	// 获取输出指针
	float* output_1 = output_tensors_1.front().GetTensorMutableData<float>();
	Mat proba_map_1 = Mat(image_size_db, image_size_db, 5, output_1);  //5:CV_32_FC1
	//开操作过滤噪声区域，并打开可能连接的字符
	Mat kernel_1 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	erode(proba_map_1, proba_map_erode_1, kernel_1);
	dilate(proba_map_erode_1, proba_map_dilate_1, kernel_1);
	Mat result_image_1 = getMaskCoordinate(proba_map_dilate_1, img);

	Mat img_pad_2, img_float_2, proba_map_erode_2, proba_map_dilate_2;
	copyMakeBorder(result_image_1, img_pad_2, 0, 160, 0, 0, cv::BORDER_CONSTANT, Scalar(0));
	img_pad_2.convertTo(img_float_2, CV_32FC3, 1 / 255.0, 0.0);		//归一化
	float* image_ptr_2 = (float*)img_float_2.data;
	// 创建输入tensor
	Ort::Value input_tensor_2 = Ort::Value::CreateTensor<float>(memory_info, image_ptr_2, input_tensor_size, input_node_dims.data(), 4);
	//运行
	auto output_tensors_2 = session_2.Run(Ort::RunOptions{ nullptr }, input_node_name_2.data(), &input_tensor_2, 1, output_node_name_2.data(), 1);
	//获取输出
	float* output_2 = output_tensors_2.front().GetTensorMutableData<float>();
	Mat proba_map_2 = Mat(image_size_db, image_size_db, 5, output_2);  //5:CV_32_FC1
	//开操作过滤噪声区域，并打开可能连接的字符
	Mat kernel_2 = getStructuringElement(MORPH_ELLIPSE, Size(17, 17));
	erode(proba_map_2, proba_map_erode_2, kernel_2);
	dilate(proba_map_erode_2, proba_map_dilate_2, kernel_2);
	std::vector<Mat> result_images = getMaskCoordinate3(proba_map_dilate_2, result_image_1);
	std::string ocr_str;
	for (int i = 0; i < result_images.size(); i++) {

		Mat img_resized_3, img_float_3;

		//imshow("result_images[i]", result_images[i]);
		//waitKey();
		resize(result_images[i], img_resized_3, Size(image_size_dec, image_size_dec));
		img_resized_3.convertTo(img_float_3, CV_32FC3, 1 / 255.0, 0.0);		//归一化
		float* image_ptr_3 = (float*)img_float_3.data;
		// 创建输入tensor
		Ort::Value input_tensor_3 = Ort::Value::CreateTensor<float>(memory_info, image_ptr_3, input_tensor_size_dec, input_node_dims_dec.data(), 4);
		//运行
		auto output_tensors_3 = session_3.Run(Ort::RunOptions{ nullptr }, input_node_name_dec.data(), &input_tensor_3, 1, output_node_name_dec.data(), 1);
		//获取输出
		int* output_3 = output_tensors_3.front().GetTensorMutableData<int>();
		ocr_str += label_map[*output_3];

	}


	//std::cout << "Decision out: " << ocr_str << std::endl;
	auto t_end_1 = std::chrono::high_resolution_clock::now();
	float total_1 = std::chrono::duration<float, std::milli>(t_end_1 - t_start_1).count();
	std::cout << image_path << std::endl;
	std::cout << "Inference takes: " << total_1 << " ms" << std::endl;
	putText(img, ocr_str.substr(0, 5), Point(100, 200), 1, 8.0, (0, 0, 255), 3);
	putText(img, ocr_str.substr(5, 10), Point(100, 300), 1, 8.0, (0, 0, 255), 3);
	Mat output_img;
	resize(img, output_img, Size(640, 320));
	imshow("result", output_img);
	waitKey();
	destroyWindow("result");
	return ocr_str;
}

std::string OcrClass::inference(void* pbuffer) {

	auto t_start_1 = std::chrono::high_resolution_clock::now();
	//读图加预处理
	Mat image = Mat(1940, 2588, 0, pbuffer);

	Mat img_float_1, img_pad_1, img_resized, img, proba_map_erode_1, proba_map_dilate_1;
	resize(image, img, Size(2048, 1536));
	double resize_scale = double(image_size_db) / double(img.size().width);
	resize(img, img_resized, Size(), resize_scale, resize_scale);
	//resize(img, img_resized, img_size);
	//Size img_size = Size(320, 240);
	int pad_size = image_size_db - img.size().height * resize_scale;
	copyMakeBorder(img_resized, img_pad_1, 0, pad_size, 0, 0, cv::BORDER_CONSTANT, Scalar(0));	//pad至目标大小
	img_pad_1.convertTo(img_float_1, CV_32FC3, 1 / 255.0, 0.0);		//归一化
	float* image_ptr_1 = (float*)img_float_1.data;

	// 创建输入tensor
	Ort::Value input_tensor_1 = Ort::Value::CreateTensor<float>(memory_info, image_ptr_1, input_tensor_size, input_node_dims.data(), 4);
	// 运行阶段1
	auto output_tensors_1 = session_1.Run(Ort::RunOptions{ nullptr }, input_node_name_1.data(), &input_tensor_1, 1, output_node_name_1.data(), 1);
	// 获取输出指针
	float* output_1 = output_tensors_1.front().GetTensorMutableData<float>();
	Mat proba_map_1 = Mat(image_size_db, image_size_db, 5, output_1);  //5:CV_32_FC1
	//开操作过滤噪声区域，并打开可能连接的字符
	Mat kernel_1 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	erode(proba_map_1, proba_map_erode_1, kernel_1);
	dilate(proba_map_erode_1, proba_map_dilate_1, kernel_1);
	Mat result_image_1 = getMaskCoordinate(proba_map_dilate_1, img);

	Mat img_pad_2, img_float_2, proba_map_erode_2, proba_map_dilate_2;
	copyMakeBorder(result_image_1, img_pad_2, 0, 160, 0, 0, cv::BORDER_CONSTANT, Scalar(0));		//pad至目标大小
	img_pad_2.convertTo(img_float_2, CV_32FC3, 1 / 255.0, 0.0);		//归一化
	float* image_ptr_2 = (float*)img_float_2.data;
	// 创建输入tensor
	Ort::Value input_tensor_2 = Ort::Value::CreateTensor<float>(memory_info, image_ptr_2, input_tensor_size, input_node_dims.data(), 4);
	//运行阶段2
	auto output_tensors_2 = session_2.Run(Ort::RunOptions{ nullptr }, input_node_name_2.data(), &input_tensor_2, 1, output_node_name_2.data(), 1);
	//获取输出
	float* output_2 = output_tensors_2.front().GetTensorMutableData<float>();
	Mat proba_map_2 = Mat(image_size_db, image_size_db, 5, output_2);  //5:CV_32_FC1
	//开操作过滤噪声区域，打开可能连接的字符
	Mat kernel_2 = getStructuringElement(MORPH_ELLIPSE, Size(17, 17));
	erode(proba_map_2, proba_map_erode_2, kernel_2);
	dilate(proba_map_erode_2, proba_map_dilate_2, kernel_2);
	std::vector<Mat> result_images = getMaskCoordinate3(proba_map_dilate_2, result_image_1);
	std::string ocr_str;
	for (int i = 0; i < result_images.size(); i++) {

		Mat img_resized_3, img_float_3;

		//imshow("result_images[i]", result_images[i]);
		//waitKey();
		resize(result_images[i], img_resized_3, Size(image_size_dec, image_size_dec));
		img_resized_3.convertTo(img_float_3, CV_32FC3, 1 / 255.0, 0.0);		//归一化
		float* image_ptr_3 = (float*)img_float_3.data;
		// 创建输入tensor
		Ort::Value input_tensor_3 = Ort::Value::CreateTensor<float>(memory_info, image_ptr_3, input_tensor_size_dec, input_node_dims_dec.data(), 4);
		//运行阶段3
		auto output_tensors_3 = session_3.Run(Ort::RunOptions{ nullptr }, input_node_name_dec.data(), &input_tensor_3, 1, output_node_name_dec.data(), 1);
		//获取输出
		int* output_3 = output_tensors_3.front().GetTensorMutableData<int>();
		ocr_str += label_map[*output_3];

	}

	//std::cout << "Decision out: " << ocr_str << std::endl;
	auto t_end_1 = std::chrono::high_resolution_clock::now();
	float total_1 = std::chrono::duration<float, std::milli>(t_end_1 - t_start_1).count();
	std::cout << "Inference takes: " << total_1 << " ms" << std::endl;
	putText(img, ocr_str, Point(100, 200), 1, 8.0, (0, 0, 255), 3);
	//Mat output_img;
	//resize(img, output_img, Size(640, 320));
	//imshow("result", output_img);
	//waitKey();
	//destroyWindow("result");
	return ocr_str;
}


Mat OcrClass::getMaskCoordinate(Mat& mask, Mat& img) {

	std::vector< std::vector< cv::Point> > contours;
	mask = mask * 255;
	Mat binary_mask, threshold_mask;
	threshold(mask, threshold_mask, 125, 255, cv::THRESH_BINARY);
	threshold_mask.convertTo(binary_mask, CV_8UC1);
	findContours(binary_mask, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	if (contours.size() > 1) {
		std::sort(contours.begin(), contours.end(), [](std::vector<Point>a, std::vector<Point>b) {if (a.size() > b.size()) return true; else return false; });
		time_t t;
		time(&t);
		struct tm* p = gmtime(&t);
		std::string filename = faultsPath + std::to_string(1900 + p->tm_year) + "-" + std::to_string(p->tm_mday) + "-" + std::to_string(p->tm_hour) + "-" + std::to_string(p->tm_min) + "-1.bmp";
		throw contours.size();		//识别到大于1个字符块，有小概率是错的
	}

	std::sort(contours[0].begin(), contours[0].end(), [](Point a, Point b) {if (a.x > b.x) return true; else return false; });
	int xmax = contours[0].front().x;
	int xmin = contours[0].back().x;

	std::sort(contours[0].begin(), contours[0].end(), [](Point a, Point b) {if (a.y > b.y) return true; else return false; });
	int ymax = contours[0].front().y;
	int ymin = contours[0].back().y;
	//int xmax = 0;
	//int xmin = 320;
	//int ymax = 0;
	//int ymin = 320;
	//for (int i = 0; i < contours[0].size(); i++) {

	//	if (contours[0][i].x > xmax) {
	//		xmax = contours[0][i].x;
	//	}
	//	if (contours[0][i].x < xmin)
	//	{
	//		xmin = contours[0][i].x;
	//	}

	//	if (contours[0][i].y > ymax) {
	//		ymax = contours[0][i].y;
	//	}
	//	if (contours[0][i].y < ymin)
	//	{
	//		ymin = contours[0][i].y;
	//	}

	//}
	float scale = float(img.size().width) / float(mask.size().width);
	int xcenter = int(((xmin + xmax) / 2) * scale);
	int ycenter = int(((ymin + ymax) / 2) * scale);

	//cv::drawContours(img, contours, -1, cv::Scalar::all(255));

	std::vector<Range> range;
	range.emplace_back(Range(ycenter - 80, ycenter + 80));
	range.emplace_back(Range(xcenter - 160, xcenter + 160));
	Mat result_image = img(range);

	//std::vector<Range> range;
	//range.emplace_back(Range(ymin * scale, ymax * scale));
	//range.emplace_back(Range(xmin * scale, xmax * scale));
	//Mat result_image = img(range);

	return result_image;

}


Mat OcrClass::getMaskCoordinate2(Mat& mask, Mat& img) {

	std::vector< std::vector< cv::Point> > contours;
	mask = mask * 255;
	float scale = 1;
	Mat binary_mask, threshold_mask;
	threshold(mask, threshold_mask, 76, 255, cv::THRESH_BINARY);
	threshold_mask.convertTo(binary_mask, CV_8UC1);

	findContours(binary_mask, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	for (int j = 0; j < contours.size(); j++) {

		cv::drawContours(img, contours, j, (0, 0, 255));
	}

	//cv::imshow("img", img);
	//cv::waitKey();
	return img;
}


std::vector<Mat> OcrClass::getMaskCoordinate3(Mat& mask, Mat& img) {

	std::vector< std::vector< cv::Point> > contours;
	mask = mask * 255;
	float scale = 1;
	Mat binary_mask, threshold_mask;
	threshold(mask, threshold_mask, 125, 255, cv::THRESH_BINARY);
	threshold_mask.convertTo(binary_mask, CV_8UC1);

	findContours(binary_mask, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	//for (int i = 0; i < contours.size(); i++) {

	//	drawContours(img, contours, i, (0, 0, 255), 3);

	//}
	//imshow("123", img);
	//waitKey();

	std::vector<std::pair<int, int>> centers;

	for (int j = 0; j < contours.size(); j++) {
		if (contours[j].size() < 10) continue;
		int xmax = 0;
		int xmin = 320;
		int ymax = 0;
		int ymin = 320;
		for (int i = 0; i < contours[j].size(); i++) {

			if (contours[j][i].x > xmax) {
				xmax = contours[j][i].x;
			}
			if (contours[j][i].x < xmin)
			{
				xmin = contours[j][i].x;
			}

			if (contours[j][i].y > ymax) {
				ymax = contours[j][i].y;
			}
			if (contours[j][i].y < ymin)
			{
				ymin = contours[j][i].y;
			}

		}

		if (ymin < 160) {
			centers.emplace_back(std::make_pair(int((xmin + xmax) / 2 * scale), int(((ymin + ymax) / 2) * scale)));
		}
	}

	if (contours.size() != 10) {
		time_t t;
		time(&t);
		struct tm* p = gmtime(&t);
		std::string filename = faultsPath + std::to_string(1900 + p->tm_year) + "-" + std::to_string(p->tm_mday) + "-" + std::to_string(p->tm_hour) + "-" + std::to_string(p->tm_min) + "-2.bmp";
		imwrite(filename, img);
		//throw contours.size();		//识别到大于10个字符，有概率是错的
	}

	std::sort(centers.begin(), centers.end(), [](std::pair<int, int>a, std::pair<int, int>b) {if (a.second < b.second) return true; else return false; });
	std::vector<std::pair<int, int>> centers_1(centers.begin(), centers.begin() + ceil(centers.size() / 2));
	std::sort(centers_1.begin(), centers_1.end(), [](std::pair<int, int>a, std::pair<int, int>b) {if (a.first < b.first) return true; else return false; });
	std::vector<std::pair<int, int>> centers_2(centers.begin() + ceil(centers.size() / 2), centers.end());
	std::sort(centers_2.begin(), centers_2.end(), [](std::pair<int, int>a, std::pair<int, int>b) {if (a.first < b.first) return true; else return false; });

	std::vector<Mat> result_image;
	int ysize = 30;
	int xsize = 24;
	for (int i = 0; i < centers_1.size(); i++) {

		std::vector<Range> range;

		range.emplace_back(Range(std::max(centers_1[i].second - ysize, 0), std::min(centers_1[i].second + ysize, 160)));
		range.emplace_back(Range(std::max(centers_1[i].first - xsize, 0), std::min(centers_1[i].first + xsize, 320)));
		result_image.emplace_back(img(range));

	}

	for (int i = 0; i < centers_2.size(); i++) {

		std::vector<Range> range;
		range.emplace_back(Range(std::max(centers_2[i].second - ysize, 0), std::min(centers_2[i].second + ysize, 160)));
		range.emplace_back(Range(std::max(centers_2[i].first - xsize, 0), std::min(centers_2[i].first + xsize, 320)));
		result_image.emplace_back(img(range));
	}

	return result_image;

}