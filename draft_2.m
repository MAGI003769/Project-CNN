clear all; close all; clc;
%输入
img=double(imread('E:\GitHub\Project-CNN\images\gray.png'));
% img=rgb2gray(img);
imshow(img,[]);
[m, n]=size(img);

%伽马校正
img=sqrt(img);

%下面是求边缘
fy=[1 2 1;
    0 0 0;
    -1 -2 -1];        %定义竖直模板
fx=-fy';             %定义水平模板

%新算子的尝试
sobel45=[-2 -1 0;
    -1  0 1;
    0  1 2];
sobel_xie=[0 1 0;
    -1 0 1;
    0 -1 0];
sobel_zhi=[1 0 1;
    0 0 0;
    -1 0 -1];
%卷积
Iy=imfilter(img,fy,'same');    %replicate or X
Ix=imfilter(img,fx,'same'); %水平边缘
G=sqrt(Ix.^2+Iy.^2);           %amplitude of gradient
s45=imfilter(img,sobel45,'same');
s_xie=imfilter(img,sobel_xie,'same');
s_zhi=imfilter(img,sobel_zhi,'same');


%求边缘强度与斜率
Ied=sqrt(Ix.^2+Iy.^2);              %边缘强度
Iphase=Iy./Ix;              %边缘斜率，有些为inf,-inf,nan，其中nan需要再处理一下

%创建八个空白图
picture_0=zeros(n);
picture_45=zeros(n);
picture_90=zeros(n);
picture_135=zeros(n);
picture_180=zeros(n);
picture_225=zeros(n);
picture_270=zeros(n);
picture_315=zeros(n);
picture_360=zeros(n);


%判定及填充
step=1;                %step滑动步长

for i=1:step:n          %如果处理的m/step不是整数，最好是i=1:step:m-step
    
    for j=1:step:n      %注释同上
        
        if isnan(Iphase(i,j))==1  %0/0会得到nan，如果像素是nan，重设为0
            Iphase(i,j)=0;
        end
        ang=atan(Iphase(i,j));    %atan求的是[-90 90]度之间
        ang=mod(ang*180/pi,360);    %全部变正，-90变270
        if Ix(i,j)<0              %根据x方向确定真正的角度
            if ang<90               %如果是第一象限
                ang=ang+180;        %移到第三象限
            end
            if ang>270              %如果是第四象限
                ang=ang-180;        %移到第二象限
            end
        end
        
        %分解向量大小
        a1= abs(Ix(i,j)-Iy(i,j)./Ied(i,j));
        a2=(sqrt(2).* min(Ix(i,j),Iy(i,j)))./Ied(i,j);
        
        
        %判别方向并填充
        if 0<ang && ang<=45 %在第一个方向中
            picture_0(i,j)=a2;
            picture_45(i,j)=a1;
        end
        if 45< ang && ang <=90
            picture_45(i,j)=a1;
            picture_90(i,j)=a2;
        end
        if 90<ang && ang<=135
            picture_90(i,j)=a2;
            picture_135(i,j)=a1;
        end
        if 135<ang && ang<=180
            picture_135(i,j)=a1;
            picture_180(i,j)=a2;
        end
        if 180<ang && ang<=225
            picture_180(i,j)=a2;
            picture_225(i,j)=a1;
        end
        if 225<ang && ang<=270
            picture_225(i,j)=a1;
            picture_270(i,j)=a2;
        end
        if 270<ang && ang<=315
            picture_270(i,j)=a2;
            picture_315(i,j)=a1;
        end
        if 315< ang && ang<=360
            picture_315(i,j)=a1;
            picture_360(i,j)=a2;
        end
        
    end
    
end

%清晰图像
Iy=255*Iy./max(Iy(:));
Ix=255*Ix./max(Ix(:));
G=255*G./max(G(:));
s45=255*s45./max(s45(:));
s_xie=255*s_xie./max(s_xie(:));
s_zhi=255*s_zhi./max(s_zhi(:));

%显示图像
subplot(3,3,1);
imshow(uint8(Ix))
title('Ix');

subplot(3,3,2);
imshow(uint8(Iy))
title('Iy');

picture_45=255*picture_45./max(picture_45(:));
subplot(3,3,3);
imshow(uint8(picture_45))
title('45');

picture_135=255*picture_135./max(picture_135(:));
subplot(3,3,4);
imshow(uint8(picture_135))
title('135');

picture_225=255*picture_225./max(picture_225(:));
subplot(3,3,5);
imshow(uint8(picture_225))
title('225');

subplot(3,3,6);
imshow(uint8(G));
title('G');

s45=255*s45./max(s45(:));
subplot(3,3,7);
imshow(uint8(s45))
title('sobel45');

s_xie=255*s_xie./max(s_xie(:));
subplot(3,3,8);
imshow(uint8(s_xie))
title('s_xie');

s_zhi=255*s_zhi./max(s_zhi(:));
subplot(3,3,9);
imshow(uint8(s_zhi))
title('s_zhi');
