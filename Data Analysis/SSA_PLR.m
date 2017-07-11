clc
clear all;
close all;

%import right diameters

filename = '<Path>/right.txt';
delimiter = {''};
formatSpec = '%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
diamDx = [dataArray{1:end-1}];
clearvars filename delimiter formatSpec fileID dataArray ans;

%import left diameters
filename = '<Path>/left.txt';
delimiter = {''};
formatSpec = '%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
diamSx = [dataArray{1:end-1}];
clearvars filename delimiter formatSpec fileID dataArray ans;

%SSA LEFT_______________________________________________________

% Set general Parameters

M = 80;    % window length = embedding dimension  LENGHT = SAMPLING FREQUENCY
N = length(diamSx);   % length
t = (1:N)';
X = diamSx;

%Calculate covariance matrix (trajectory approach)

Y=zeros(N-M+1,M);
for m=1:M
    Y(:,m) = X((1:N-M+1)+m-1);
end;
Cemb=Y'*Y / (N-M+1);

%Choose covariance estimation

C=Cemb;

%Calculate eigenvalues LAMBDA and eigenvectors RHO

[RHO,LAMBDA] = eig(C);
LAMBDA = diag(LAMBDA);               % extract the diagonal elements
[LAMBDA,ind]=sort(LAMBDA,'descend'); % sort eigenvalues
RHO = RHO(:,ind);                    % and eigenvectors

%Calculate principal components PC

PC = Y*RHO;

%Calculate reconstructed components RC

RC=zeros(N,M);
for m=1:M
    buf=PC(:,m)*RHO(:,m)'; % invert projection
    buf=buf(end:-1:1,:);
    for n=1:N % anti-diagonal averaging
        RC(n,m)=mean( diag(buf,-(N-M+1)+n) );
    end
end;

     figure;
     set(gcf,'name','Signal recostruction: left eye')
     clf;
     plot(t,X,'g*',t,sum(RC(:,1:3),2),'b', 'LineWidth',2),xlabel('Frames'),ylabel('Diameters (mm)'),title('Signal recostruction: left eye'), grid on;
     legend('Original','Reconstructed Components');

XSX = sum(RC(:,1:3),2);
tSX = t;

gradXSX = gradient(XSX, 0.016);
lMAX = min(XSX)
lMIN = max(XSX)
lCH = 100*(lMIN - lMAX)/lMAX
lMCV = abs(min(gradXSX))


%SSA LEFT_______________________________________________________

% Set general Parameters

M = 80;    % window length = embedding dimension
N = length(diamDx);   % length
t = (1:N)';
X = diamDx;

%Calculate covariance matrix (trajectory approach)

Y=zeros(N-M+1,M);
for m=1:M
    Y(:,m) = X((1:N-M+1)+m-1);
end;
Cemb=Y'*Y / (N-M+1);

%Choose covariance estimation

C=Cemb;

%Calculate eigenvalues LAMBDA and eigenvectors RHO

[RHO,LAMBDA] = eig(C);
LAMBDA = diag(LAMBDA);               % extract the diagonal elements
[LAMBDA,ind]=sort(LAMBDA,'descend'); % sort eigenvalues
RHO = RHO(:,ind);                    % and eigenvectors

%Calculate principal components PC

PC = Y*RHO;

%Calculate reconstructed components RC

RC=zeros(N,M);
for m=1:M
    buf=PC(:,m)*RHO(:,m)'; % invert projection
    buf=buf(end:-1:1,:);
    for n=1:N % anti-diagonal averaging
        RC(n,m)=mean( diag(buf,-(N-M+1)+n) );
    end
end;
XDX = sum(RC(:,1:3),2)
     figure;
     set(gcf,'name','Signal recostruction: right eye')
     clf;
     plot(t,X,'g*',t, XDX,'b', 'LineWidth',2),xlabel('Frames'),ylabel('Diameters (mm)'),title('Signal recostruction: right eye'), grid on;
     legend('Original','Reconstructed Components');

tDX = t;

gradXDX = gradient(XDX, 0.016);
rMAX = min(XDX)
rMIN = max(XDX)
rCH = 100*(rMIN - rMAX)/rMAX
rMCV = abs(min(gradXDX))


%plot the filtered diameters together, both left and right eye
figure, plot(tSX,XSX, 'b',tDX, XDX, 'g', 'Linewidth',2),legend('left','right'),xlabel('Frames'),ylabel('Diameters (mm)'),title('PLR test'), grid on;


%calculate gradients

% plot gradient
figure, plot(tSX,gradXSX, 'b',tDX,gradXDX,'g', 'Linewidth',2),legend('left','right'),xlabel('Frames'),ylabel('Speed (mm/sec)'),title('PLR test: gradients'), grid on;


