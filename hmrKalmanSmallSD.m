%This function perform the Kalman Filter to estimate the HRF by using a set
%of temporal basis functions (gaussian are used here). It also uses a small
%separation measurement as a regressor with an AR process (Zhang et al). We
%thus have a simultaneous dynamical HRF estimation/regression. This
%function doesn't perform the block average. We run the filter two times
%and the second time we use the final state vales and state covariance as
%the initial values. We finally run a Smoother to avoid phase delay in the 
%recovered response.

%created 8-06-2010 by L. Gagnon
%modified 9-24-2010 by L. Gagnon (fixed a typo with initialization of state
%covariance matrix)

function [ddf ddfS Pfinal PfinalS] = hmrKalmanSmallSD(dd, ddref, onset_vect, t, maxTHRF, ntap, iniCov, ProcNoise, MeasNoise, HRFest)

global tbasis

%temporal basis function (gaussian with std of 1.5 sec and their 
%means are separated by 1.5 sec)

gstd=0.5; %std of 0.5 sec
gms=0.5; %mean separated by 0.5 sec

%basis function and timing
nB=floor(maxTHRF/gms)-1;
dt=t(2)-t(1);
nT=length(t);
tb=(0:dt:maxTHRF)';
nTb=length(tb);
tbasis=zeros(nTb,nB);
for b=1:nB
    tbasis(:,b)=exp(-(tb-b*gms).^2/(2*gstd.^2));
    tbasis(:,b)=tbasis(:,b)./max(tbasis(:,b)); %normalize to 1
end

%construct u-vector
u=[];
for i=1:nB
    u(:,i)=conv(onset_vect,tbasis(:,i));
end
u=u(1:nT,1:nB);

%normalize time course so that their std is 1
scale_dd=std(dd);
scale_ddref=std(ddref);
dd(:,1)=dd(:,1)./scale_dd(1,1);
% dd(:,2)=dd(:,2)./scale_dd(1,2);
ddref(:,1)=ddref(:,1)./scale_ddref(1,1);
% ddref(:,2)=ddref(:,2)./scale_ddref(1,2);

%least-square fit setting
tRangeHRF=[0 maxTHRF];
Xo=[-100*ones(nB,1) ones(nB,1) 100*ones(nB,1)];
options = optimset('TolFun',1E-6,'TolX',1E-6,'MaxFunEvals',2E4,'MaxIter',2E4,'display','off',...
    'Diagnostics','off'); %new algorithm (with the reflective boundary stuff)
xo=zeros(nB,2);

ddf=zeros(nT,1);
ddfS=zeros(nT,1);
aaa=length(tb);bbb=length(HRFest);
if (bbb>aaa)
    HRFest=HRFest(1:aaa,:);
end

for conc=1%:2

%least-square fit to the simultaneous deconvolution to get the initial weigths
[xo(:,conc),resnorm,residual,exitflag,output,lambda] = lsqcurvefit(@tbasis_fit,Xo(:,2),tb,HRFest(:,conc)./scale_dd(1,conc),Xo(:,1),Xo(:,3),options);

%initialize state vector
nS=nB+ntap;
x=zeros(nS,nT); %X_k|k
xp=zeros(nS,nT); %X_k|k-1 (need to store that for the smoother)
xN=zeros(nS,nT); %X_k|N output of the smoother
x(:,1)=[ xo(:,conc); 1; zeros(nS-nB-1,1) ];
xp(:,1)=x(:,1);

%intialize process noise matrix
if length(ProcNoise)==1
    Q=diag(ProcNoise.*ones(nS,1));
elseif length(ProcNoise)==2
    Q=diag([ProcNoise(1).*ones(1,nB) ProcNoise(2).*ones(1,ntap)]);
end

%initialize observation matrix
C=zeros(nT,nS);
C(1,:)=[u(1,:) ddref(1,conc) zeros(1,ntap-1)];

%intialize measurement noise matrix
R=diag(MeasNoise);

%initialize covariance matrix
P=zeros(nS,nS,nT); %P_k|k
Pp=zeros(nS,nS,nT); %P_k|k-1 (need to store that for the smoother)
PN=zeros(nS,nS,nT); %P_k|N output of the smoother
if length(iniCov)==1
    P(:,:,1)=diag(iniCov*ones(nS,1));
elseif length(iniCov)==2
    P(:,:,1)=diag([iniCov(1).*ones(1,nB) iniCov(2).*ones(1,ntap)]);
else %(we have the complete matrix)
    P(:,:,1)=iniCov;
end
Pp(:,:,1)=P(:,:,1);


%Kalman Filter
for k=2:nT

    %define C matrix for this time step
    if k<ntap
        C(k,:)=[u(k,:) ddref(k:-1:1,conc)' zeros(1,ntap-k)];
    elseif k>=ntap
        C(k,:)=[u(k,:) ddref(k:-1:k-ntap+1,conc)'];
    end

    %Kalman Filter
    xp(:,k) = x(:,k-1); %state update
    Pp(:,:,k) = P(:,:,k-1)+Q;%covariance update
    K = Pp(:,:,k)*C(k,:)'/( C(k,:)*Pp(:,:,k)*C(k,:)' + R ); %Kalman Gain
    x(:,k) = xp(:,k) + K*( dd(k,conc) - C(k,:)*xp(:,k) ); %update state
    P(:,:,k) = ( eye(nS) - K*C(k,:) )*Pp(:,:,k); %update covariance    

    
end %k

%run Smoother
xN(:,nT)=x(:,nT);
PN(:,:,nT)=P(:,:,nT);
for k=nT-1:-1:1

    xN(:,k) = x(:,k) + P(:,:,k)*Pp(:,:,k+1)*( xN(:,k+1) - xp(:,k+1) ); %update state
    PN(:,:,k) = P(:,:,k) + P(:,:,k)*Pp(:,:,k+1)*( PN(:,:,k+1) - Pp(:,:,k+1) ); %update covariance
    
end

%final covariance
Pfinal=P(:,:,nT);
PfinalS=PN(:,:,1);

%compute final HRF based on state and rescale
for k=1:nT
    ddf(k,conc) = scale_dd(1,conc).*C(k,1:nB)*x(1:nB,k); %NOT using Smoother
end %k
for k=1:nT
    ddfS(k,conc) = scale_dd(1,conc).*C(k,1:nB)*xN(1:nB,k); %using Smoother
end %k

end %conc

clear tbasis; %since it is global

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yhat = tbasis_fit(params,t)

global tbasis

w=params;
nT=length(t);
yhat=zeros(nT,1);
for i=1:length(w)
    yhat=yhat+w(i).*tbasis(:,i);
end


return


