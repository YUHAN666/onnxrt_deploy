#include "inference.h"

const wchar_t* model_path_1 = L"./models/model_1.onnx";
const wchar_t* model_path_2 = L"./models/model_2.onnx";
const wchar_t* model_path_3 = L"./models/model_dec.onnx";
const char* image_path = "./data";
const char* extention = ".bmp";

void GetFiles(std::string path, std::vector<std::string>& fileNames, std::string extention)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;

	if ((hFile = _findfirst(p.assign(path).append("/*" + extention).c_str(), &fileinfo)) != -1)
	{
		do
		{
			std::string q;
			q.append(path).append("/").append(fileinfo.name);
			fileNames.push_back(q);
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}


int main(int argc, char* argv[]) {

	std::vector<std::string> filenames;
	// ��ȡͼƬĿ¼
	GetFiles(image_path, filenames, extention);
	std::string fault_path = "./faults/";

	OcrClass* ocr = new OcrClass(model_path_1, model_path_2, model_path_3, fault_path);

	for (int i = 0; i < filenames.size(); i++) {
		ocr->inference(filenames[i].c_str());
	}

}