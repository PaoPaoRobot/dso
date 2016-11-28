/**
* This file is part of DSO.
*
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#pragma once
#include "util/settings.h"
#include "util/NumType.h"

namespace dso
{
extern int wG[PYR_LEVELS], hG[PYR_LEVELS];       ///< 图像的宽度和高度
extern float fxG[PYR_LEVELS], fyG[PYR_LEVELS],   ///< 相机焦距
       cxG[PYR_LEVELS], cyG[PYR_LEVELS];          ///< 光心

extern float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS], ///< 逆焦距
       cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];        ///< 逆光心

extern Eigen::Matrix3f KG[PYR_LEVELS],KiG[PYR_LEVELS]; ///< 相机矩阵

extern float wM3G; // = w -3; w Minus 3, Global
extern float hM3G;

void setGlobalCalib ( int w, int h, Eigen::Matrix3f K );
}
