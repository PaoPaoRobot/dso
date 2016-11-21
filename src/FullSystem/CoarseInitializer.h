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

#include "util/NumType.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"
#include "util/settings.h"
#include "vector"
#include <math.h>




namespace dso
{
struct CalibHessian;
struct FrameHessian;


// 像素点？
struct Pnt {
public:
    // index in jacobian. never changes (actually, there is no reason why).
    float u,v;

    // idepth / isgood / energy during optimization.
    float idepth;
    bool isGood;
    Vec2f energy;		// (UenergyPhotometric, energyRegularizer)
    bool isGood_new;
    float idepth_new;
    Vec2f energy_new;

    float iR;           ///< 这个是做啥的?
    float iRSumNum;

    float lastHessian;
    float lastHessian_new;

    // max stepsize for idepth (corresponding to max. movement in pixel-space).
    float maxstep;

    // idx (x+y*w) of closest point one pyramid level above.
    int parent;
    float parentDist;

    // idx (x+y*w) of up to 10 nearest points in pixel space.
    int neighbours[10];
    float neighboursDist[10];

    float my_type;
    float outlierTH;
};

// 初始化
class CoarseInitializer
{
public:
    CoarseInitializer ( int w, int h );
    ~CoarseInitializer();


    void setFirst (	CalibHessian* HCalib, FrameHessian* newFrameHessian );
    bool trackFrame ( FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps );
    void calcTGrads ( FrameHessian* newFrameHessian );

    int frameID;
    bool fixAffine;
    bool printDebug;

    Pnt* points[PYR_LEVELS];      ///< on firstFrmame
    int numPoints[PYR_LEVELS];
    AffLight thisToNext_aff;
    SE3 thisToNext;


    FrameHessian* firstFrame;
    FrameHessian* newFrame;
private:
    Mat33 K[PYR_LEVELS];
    Mat33 Ki[PYR_LEVELS];
    double fx[PYR_LEVELS];
    double fy[PYR_LEVELS];
    double fxi[PYR_LEVELS];
    double fyi[PYR_LEVELS];
    double cx[PYR_LEVELS];
    double cy[PYR_LEVELS];
    double cxi[PYR_LEVELS];
    double cyi[PYR_LEVELS];
    int w[PYR_LEVELS];
    int h[PYR_LEVELS];
    
    // 计算每层金字塔的内参
    void makeK ( CalibHessian* HCalib );

    bool snapped;
    int snappedAt;

    // pyramid images & levels on all levels
    Eigen::Vector3f* dINew[PYR_LEVELS];
    Eigen::Vector3f* dIFist[PYR_LEVELS];

    Eigen::DiagonalMatrix<float, 8> wM;

    // temporary buffers for H and b.
    Vec10f* JbBuffer;			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
    Vec10f* JbBuffer_new;

    Accumulator9 acc9;
    Accumulator9 acc9SC;


    Vec3f dGrads[PYR_LEVELS];

    float alphaK;
    float alphaW;
    float regWeight;
    float couplingWeight;

    /**
     * @brief calculates residual, Hessian and Hessian-block needed for re-substituting depth.
     * @param 指定层
     * @param H_out
     * @param b_out
     * @param H_out_sc
     * @param b_out_sc
     * @param refToNew
     * @param refToNew_aff
     * @param plot
     * @return
     */
    Vec3f calcResAndGS (
        int lvl,
        Mat88f &H_out, Vec8f &b_out,
        Mat88f &H_out_sc, Vec8f &b_out_sc,
        SE3 refToNew, AffLight refToNew_aff,
        bool plot );
    Vec3f calcEC ( int lvl ); // returns OLD NERGY, NEW ENERGY, NUM TERMS.

    /**
     * @brief 算一个加权平均(根据neighbour平滑一下idepth)
     * @param 层数
     */
    void optReg ( int lvl );

    /**
     * @brief 把当src层的idepth上传
     * @param src层
     */
    void propagateUp ( int srcLvl );

    /**
     * @brief 把当src层的idepth下传
     * @param src层
     */
    void propagateDown ( int srcLvl );
    float rescale();

    /**
     * @brief 把idepth初始化进行完毕,并设置energy=0
     * @param 指定层
     */
    void resetPoints ( int lvl );
    void doStep ( int lvl, float lambda, Vec8f inc );
    void applyStep ( int lvl );

    void makeGradients ( Eigen::Vector3f** data );

    void debugPlot ( int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps );
    void makeNN();
};


struct FLANNPointcloud {
    inline FLANNPointcloud() {
        num=0;
        points=0;
    }
    inline FLANNPointcloud ( int n, Pnt* p ) :  num ( n ), points ( p ) {}
    int num;
    Pnt* points;
    inline size_t kdtree_get_point_count() const {
        return num;
    }

    /**
     * @brief 计算距离的方法，这个函数定义比较重要，在L2_Simple_Adaptor里面有一个内部的调用
     */
    inline float kdtree_distance ( const float *p1, const size_t idx_p2,size_t /*size*/ ) const {
        const float d0=p1[0]-points[idx_p2].u;
        const float d1=p1[1]-points[idx_p2].v;
        return d0*d0+d1*d1;
    }

    inline float kdtree_get_pt ( const size_t idx, int dim ) const {
        if ( dim==0 ) {
            return points[idx].u;
        } else {
            return points[idx].v;
        }
    }
    template <class BBOX>
    bool kdtree_get_bbox ( BBOX& /* bb */ ) const {
        return false;
    }
};

}


