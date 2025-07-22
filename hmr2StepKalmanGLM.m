
function [ddK1 ddKS1 Pfinal PfinalS] = hmrKalmanGLMAutoINI_2step(d, s, t, SD, threshold, trange , iniCov, ProcNoise)

% UI NAME: 
% hmrKalmanGLMAutoINI_2step
% Kalman_GLM_AutoINI_2step  created 05-2020 by S. Jahani  
% Kalman_GLM
% This script estimates the HRF in two steps using Kalman filter and GLM with Automatic
% intialization for measurement noise
%
% INPUTS:
% d - intensity data (#time points x #data channels
% s - stimulation vector (# time points x #conditions)=1 at stim onset otherwise =0
% t - time vector corresponding with d and s
% threshold - short separation distance Threshold (units should be consistent with SD)
% SD - source detector stucture (units should be consistent with threshold)
% trange - defines the range for the block average [tPre tPost]
% ProcNoise - Kalman Filter process noise (how much you trust the model and the initial estimates)
% iniCov - initial estimate of the covariance
%
% OUTPUTS:
% ddK1 - final HRF - No smoothing
% ddKS1 - final HRF Using - smoothing
% Pfinal - final covariance - No smoothing
% PfinalS - final covariance - Using smoothing
%

ppf=[6 6]; %partial pathlength correction
hpf=0; %High pass filter
lpf=0.5; %Low pass filter
dod = hmrIntensity2OD(d); %Intensity to optical density
dConc = hmrIntensity2Conc_new(d,SD,t,hpf,1.25,ppf); % optical density to concentration
% ProcNoise=[2.5e-6 5e-6]; %Kalman Filter process noise (how much you trust the model and the initial estimates)
% iniCov=[1e-6 5e-6]; %initial estimate of the covariance
MeasNoise = cov(dod(:,iCh)); % measurement noise (how much you trust the measurements)

%Kalman Filter measurement noise (how much you trust the measurements)
threshold = 12;  % mm (12)/cm (1.2)!! look your SD units short sep - source distance
nMeas=length(find(SD.MeasList(:,4)==1));
%compute distance for each measurement (in cm)
Distances=((SD.SrcPos(SD.MeasList(:,1),1) - SD.DetPos(SD.MeasList(:,2),1)).^2 +...
    (SD.SrcPos(SD.MeasList(:,1),2) - SD.DetPos(SD.MeasList(:,2),2)).^2 +...
    (SD.SrcPos(SD.MeasList(:,1),3) - SD.DetPos(SD.MeasList(:,2),3)).^2).^0.5;
Distances=Distances(1:nMeas);

%position of small separation
small_pos_idx=find(Distances<threshold);
long_pos_idx=find(Distances>threshold);


corrBase_task(iCh,1,ii) = corr(dConc(:,1,iCh),dConc(:,1,isCh)).^2; %HbO
[foo_task corrIdxHbO_task] = sort(corrBase_task(iCh,1,:),3,'descend');
bestCorrHbO_task = small_pos_idx(corrIdxHbO_task(1));
dc_ref=[squeeze(dConc(:,1,bestCorrHbO_task))];


maxTHRF=trange(end);
fs=1/(t(2)-t(1)); ntap=1;
[davg, tHRF, b, nTrials] = hmrDeconvAuxTB(dConc, dc_ref, s, t, trange, ntap);
HRFest = hmrBandpassFilt( davg, fs, 0, 0.5 );
global tbasis

%temporal basis function (gaussian with std of gstd sec and their
%means are separated by gms sec)

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
scale_ddref=std(dc_ref);
dd(:,1)=dd(:,1)./scale_dd(1,1);
dc_ref(:,1)=dc_ref(:,1)./scale_ddref(1,1);

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
    C(1,:)=[u(1,:) dc_ref(1,conc) zeros(1,ntap-1)];
    
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
            C(k,:)=[u(k,:) dc_ref(k:-1:1,conc)' zeros(1,ntap-k)];
        elseif k>=ntap
            C(k,:)=[u(k,:) dc_ref(k:-1:k-ntap+1,conc)'];
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
    
    
    for k=1:nT
        ddf(k,conc) = scale_dd(1,conc).*(dd(k,conc)-C(k,end)*x(end,k)); %NOT using Smoother
    end %k
    for k=1:nT
        ddfS(k,conc) = scale_dd(1,conc).*(dd(k,conc)-C(k,end)*xN(end,k)); %using Smoother
    end %k
    
end %conc

MeasNoise=cov(ddfS);
%created 8-06-2010 by L. Gagnon
[ddK1 ddKS1 Pfinal1 PfinalS1] = hmrKalmanSmallSD(ddfS, C, s, t, maxTHRF, ntap, Pfinal1, ProcNoise, MeasNoise, HRFest); %run 2nd time with final covariance estimate

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

