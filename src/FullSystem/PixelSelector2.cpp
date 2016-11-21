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


#include "FullSystem/PixelSelector2.h"

// 



#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso
{


PixelSelector::PixelSelector(int w, int h)
{
    randomPattern = new unsigned char[w*h];
    std::srand(3141592);	// want to be deterministic.
    for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;

    currentPotential=3;


    gradHist = new int[100*(1+w/32)*(1+h/32)];
    ths = new float[(w/32)*(h/32)+100];
    thsSmoothed = new float[(w/32)*(h/32)+100];

    allowFast=false;
    gradHistFrame=0;
}

PixelSelector::~PixelSelector()
{
    delete[] randomPattern;
    delete[] gradHist;
    delete[] ths;
    delete[] thsSmoothed;
}

/**
 * @brief 计算直方图的值到多少的时候正好超过某百分比
 * @param 直方图
 * @param 折扣量
 * @return 刚好超过时候直方图的值
 */
int computeHistQuantil(int* hist, float below)
{
    int th = hist[0]*below+0.5f;  ///< 自适应的阈值，注意直方图的第一个参数是总个数，然后below等于0.5
    for(int i=0;i<90;i++)
    {
        th -= hist[i+1];
        if(th<0) return i;
    }
    return 90;
}


void PixelSelector::makeHists(const FrameHessian* const fh)
{
    gradHistFrame = fh;
    float * mapmax0 = fh->absSquaredGrad[0];

    int w = wG[0];
    int h = hG[0];

    int w32 = w/32;
    int h32 = h/32;
    thsStep = w32;


    /**
     *  mapmax0
     *     |_|_|_|_|_|
     *     |_|_|#|_|_|
     *     |_|_|_|_|_|
     *     |_|_|_|_|_|
     * 上图中每个小块大小为32 * 32
     * 当x = 3, y = 2时，map0指针指向#块的首地址
     * 这时候注意hist0指针有50个空间如下：
     * |_|__|__|__|__|**|__|__|...|__|__|__|
     *  # 00 01 02 03 04 05 06 ... 47 48 49
     * #号表示的是总的计数器，其他位表示的是梯度值，假如这个小块块像素梯度为4时，**处值加１
     */
    for(int y=0;y<h32;y++)
        for(int x=0;x<w32;x++)
        {
            float* map0 = mapmax0+32*x+32*y*w;
            int* hist0 = gradHist;// + 50*(x+y*w32);　///< 注意，这个gradHist多分配了空间，那些空间并未使用(可能在其他版本有使用)
            memset(hist0,0,sizeof(int)*50);

            for(int j=0;j<32;j++) for(int i=0;i<32;i++)
            {
                int it = i+32*x;
                int jt = j+32*y;
                if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;
                int g = sqrtf(map0[i+j*w]);
                if(g>48) g=48;
                hist0[g+1]++;
                hist0[0]++;
            }

            ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd; ///< 得到这个像素块的"直方图特征"
        }

    /**
     *　做平滑处理
     */
    for(int y=0;y<h32;y++)
        for(int x=0;x<w32;x++)
        {
            float sum=0,num=0;
            if(x>0)
            {
                if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
                if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
                num++; sum+=ths[x-1+(y)*w32];
            }

            if(x<w32-1)
            {
                if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
                if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
                num++; sum+=ths[x+1+(y)*w32];
            }

            if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
            if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
            num++; sum+=ths[x+y*w32];

            thsSmoothed[x+y*w32] = (sum/num) * (sum/num);


            if(setting_fixGradTH > 0)
                thsSmoothed[x+y*w32] = setting_fixGradTH*setting_fixGradTH;
        }





}


int PixelSelector::makeMaps(
        const FrameHessian* const fh,
        float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
    float numHave=0;
    float numWant=density;
    float quotia;
    int idealPotential = currentPotential;

    {
        // the number of selected pixels behaves approximately as
        // K / (pot+1)^2, where K is a scene-dependent constant.
        // we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

        if(fh != gradHistFrame) makeHists(fh);

        // select!
        Eigen::Vector3i n = this->select(fh, map_out,currentPotential, thFactor); ///< 选择梯度好的像素点

        // sub-select!
        numHave = n[0]+n[1]+n[2];
        quotia = numWant / numHave;

        /**
         * 以下那段递归调用的代码在调整需要采样点的个数
         * 公式从哪里来的?
         */
        // by default we want to over-sample by 40% just to be sure.
        float K = numHave * (currentPotential+1) * (currentPotential+1);
        idealPotential = sqrtf(K/numWant)-1;	// round down.
        if(idealPotential<1) idealPotential=1;

        if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
        {
            // 点太少的意思
            // re-sample to get more points!
            // potential needs to be smaller
            if(idealPotential>=currentPotential)
                idealPotential = currentPotential-1;

            currentPotential = idealPotential;
            return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);
        }
        else if(recursionsLeft>0 && quotia < 0.25)
        {
            // 点太多的意思 
            // re-sample to get less points!

            if(idealPotential<=currentPotential)
                idealPotential = currentPotential+1;

            currentPotential = idealPotential;
            return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

        }
    }

    /**
     * 然后再次减少采样点个数，如果randomPattern的阈值大于了像素值，那么丢弃该点
     * 至此，选择完毕
     */
    int numHaveSub = numHave;
    if(quotia < 0.95)
    {
        int wh=wG[0]*hG[0];
        int rn=0;
        unsigned char charTH = 255*quotia;
        for(int i=0;i<wh;i++)
        {
            if(map_out[i] != 0)
            {
                if(randomPattern[rn] > charTH )
                {
                    map_out[i]=0;
                    numHaveSub--;
                }
                rn++;
            }
        }
    }

    currentPotential = idealPotential;


    if(plot)
    {
        int w = wG[0];
        int h = hG[0];


        MinimalImageB3 img(w,h);

        for(int i=0;i<w*h;i++)
        {
            float c = fh->dI[i][0]*0.7;
            if(c>255) c=255;
            img.at(i) = Vec3b(c,c,c);
        }
        IOWrap::displayImage("Selector Image", &img);

        for(int y=0; y<h;y++)
            for(int x=0;x<w;x++)
            {
                int i=x+y*w;
                if(map_out[i] == 1)
                    img.setPixelCirc(x,y,Vec3b(0,255,0));
                else if(map_out[i] == 2)
                    img.setPixelCirc(x,y,Vec3b(255,0,0));
                else if(map_out[i] == 4)
                    img.setPixelCirc(x,y,Vec3b(0,0,255));
            }
        IOWrap::displayImage("Selector Pixels", &img);
    }

    return numHaveSub;
}


// 
Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
                                      float* map_out, int pot, float thFactor)
{

    Eigen::Vector3f const * const map0 = fh->dI;

    float * mapmax0 = fh->absSquaredGrad[0];
    float * mapmax1 = fh->absSquaredGrad[1];
    float * mapmax2 = fh->absSquaredGrad[2];


    int w = wG[0];
    int w1 = wG[1];
    int w2 = wG[2];
    int h = hG[0];

    /**
     * 请输入python代码：
     * from numpy import *
     *
     * for i in xrange(-7, 9):
     *     print(cos(pi * i / 16), sin(pi * i / 16))
     */
    // 总之是沿着圆周十六等分之后的东西 
    const Vec2f directions[16] = {
        Vec2f(0,    1.0000),
        Vec2f(0.3827,    0.9239),
        Vec2f(0.1951,    0.9808),
        Vec2f(0.9239,    0.3827),
        Vec2f(0.7071,    0.7071),
        Vec2f(0.3827,   -0.9239),
        Vec2f(0.8315,    0.5556),
        Vec2f(0.8315,   -0.5556),
        Vec2f(0.5556,   -0.8315),
        Vec2f(0.9808,    0.1951),
        Vec2f(0.9239,   -0.3827),
        Vec2f(0.7071,   -0.7071),
        Vec2f(0.5556,    0.8315),
        Vec2f(0.9808,   -0.1951),
        Vec2f(1.0000,    0.0000),
        Vec2f(0.1951,   -0.9808)};

    memset(map_out,0,w*h*sizeof(PixelSelectorStatus));


    float dw1 = setting_gradDownweightPerLevel;
    float dw2 = dw1*dw1;


    /**
     * 接下来的部分，首先是在构造对象的时候初始化了一系列随机数，用来随机选择前面directions的方向
     * 然后我们需要知道map_out里面存储的是这个点在金字塔第几层是好点(前三层给你玩)
     * 接下来是时候忙活了，pot是当前的潜在个数？，反正我查了一下，等于3,这个值还和idealPotential有关，用于决定选择的点的个数
     * 再后面就在循环了，x4 y4表示一次跳4 * pot这么多格，x3 y3表示一次跳2 * pot, x2 y2表示一次跳pot这么多
     * 之后寻址，依次对第0层，第1层，第2层的梯度值进行了操作，具体操作如下：
     * 首先是得到阈值(详见makeHists)：
     * pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];
     * pixelTH1 = pixelTH0*dw1;pixelTH2 = pixelTH1*dw2;
     * 然后计算方向导数dirNorm = fabsf((float)(agxd.dot(dir2)));
     * 最后判断哪个好，哪个好就对n几进行自加，同时在该像素点处做好标记,最后返回
     */
    int n3=0, n2=0, n4=0;
    for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
    {
        int my3 = std::min((4*pot), h-y4);
        int mx3 = std::min((4*pot), w-x4);
        int bestIdx4=-1; float bestVal4=0;
        Vec2f dir4 = directions[randomPattern[n2] & 0xF];
        for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
        {
            int x34 = x3+x4;
            int y34 = y3+y4;
            int my2 = std::min((2*pot), h-y34);
            int mx2 = std::min((2*pot), w-x34);
            int bestIdx3=-1; float bestVal3=0;
            Vec2f dir3 = directions[randomPattern[n2] & 0xF];
            for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
            {
                int x234 = x2+x34;
                int y234 = y2+y34;
                int my1 = std::min(pot, h-y234);
                int mx1 = std::min(pot, w-x234);
                int bestIdx2=-1; float bestVal2=0;
                Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
                {
                    assert(x1+x234 < w);
                    assert(y1+y234 < h);
                    int idx = x1+x234 + w*(y1+y234);
                    int xf = x1+x234;
                    int yf = y1+y234;

                    if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;


                    float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];
                    float pixelTH1 = pixelTH0*dw1;
                    float pixelTH2 = pixelTH1*dw2;


                    float ag0 = mapmax0[idx];
                    if(ag0 > pixelTH0*thFactor)
                    {
                        Vec2f ag0d = map0[idx].tail<2>();
                        float dirNorm = fabsf((float)(ag0d.dot(dir2)));
                        if(!setting_selectDirectionDistribution) dirNorm = ag0;

                        if(dirNorm > bestVal2)
                        { bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
                    }
                    if(bestIdx3==-2) continue;

                    float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
                    if(ag1 > pixelTH1*thFactor)
                    {
                        Vec2f ag0d = map0[idx].tail<2>();
                        float dirNorm = fabsf((float)(ag0d.dot(dir3)));
                        if(!setting_selectDirectionDistribution) dirNorm = ag1;

                        if(dirNorm > bestVal3)
                        { bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
                    }
                    if(bestIdx4==-2) continue;

                    float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];
                    if(ag2 > pixelTH2*thFactor)
                    {
                        Vec2f ag0d = map0[idx].tail<2>();
                        float dirNorm = fabsf((float)(ag0d.dot(dir4)));
                        if(!setting_selectDirectionDistribution) dirNorm = ag2;

                        if(dirNorm > bestVal4)
                        { bestVal4 = dirNorm; bestIdx4 = idx; }
                    }
                }

                if(bestIdx2>0)
                {
                    map_out[bestIdx2] = 1;
                    bestVal3 = 1e10;
                    n2++;
                }
            }

            if(bestIdx3>0)
            {
                map_out[bestIdx3] = 2;
                bestVal4 = 1e10;
                n3++;
            }
        }

        if(bestIdx4>0)
        {
            map_out[bestIdx4] = 4;
            n4++;
        }
    }


    return Eigen::Vector3i(n2,n3,n4);
}


}

