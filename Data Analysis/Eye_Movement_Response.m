clc
clear all
close all

fileRight = '/Users/filippobracco/Dropbox/Progetto BIE-PInCS/Dati/Stimoli mobili/destra.txt';
fileLeft = '/Users/filippobracco/Dropbox/Progetto BIE-PInCS/Dati/Stimoli mobili/sinistra.txt';
K = 0.24;
right = importdata(fileRight);
left = importdata(fileLeft);
timestep = 1/60;

xd = right(:,1);
yd = right(:,2);

xs = left(:,1);
ys = left(:,2);

d=sqrt((xd-xs).^2+(yd-ys).^2);

% smooth row data | DISTANCE
dsmooth = smoothdata(d,'gaussian',30);

%plot distance
figure,hold on,plot(1:length(dsmooth),dsmooth),xlabel('frames'),ylabel('distance [pixels]'),title('distance between eyes');
plot(1:length(d),d),hold off,legend('smoothed','raw data');

n = [1:length(xd)];

xdsmooth = smooth(xd,80,'loess');
ydsmooth = smooth(yd,80,'loess');
xssmooth = smooth(xs,80,'loess');
yssmooth = smooth(ys,80,'loess');

xdis = K*abs(xdsmooth - xssmooth);
ydis = K*abs(ydsmooth - yssmooth);

varx = var(xdis)
vary = var(ydis)

index = varx + vary

% calculate gradient on smoothed data
gradxd = gradient(xdsmooth, timestep);
gradyd = gradient(ydsmooth, timestep);
gradxs = gradient(xssmooth, timestep);
gradys = gradient(yssmooth, timestep);

gradd = sqrt(gradxd(1:500).^2 + gradyd(1:500).^2);
grads = sqrt(gradxs(1:500).^2 + gradys(1:500).^2);


%plot the gradient
figure,plot(1:length(gradd),gradd,'g','Linewidth',2),hold on,plot(1:length(grads),grads,'b','Linewidth',2),title('EM: eye gradients'),xlabel('Frames'), ylabel('Gradient (pixel/s)'),legend('Right','Left'),grid on;
%plot eyes x and y 
figure, plot(n,xd, 'g*',n,xdsmooth,'b', 'Linewidth',2),title('EM: eye x-coordinates'),xlabel('Frames'), ylabel('x-coordinate (pixel)'),legend('Data','Smoothed'),grid on;
figure, plot(n,yd, 'g*',n,ydsmooth,'b', 'Linewidth',2),title('EM: eye y-coordinates'),xlabel('Frames'), ylabel('y-coordinate (pixel)'),legend('Data','Smoothed'),grid on;


