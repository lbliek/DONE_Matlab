function [X_min,MEASUREMENTS,OPTS,MODEL] = DONE(FUN,X0,N,LB,UB,D,LAMBDA,SIGMA,EXPL)
% DONE minimizes a function using random Fourier expansions. The function
% may suffer from noise.
%
% Example: X_min = DONE(@(x) x^2+0.01*randn, 0.7, 20)
%
% X_min = DONE(FUN,X0,N) minimizes the function FUN starting from
% the initial point X0 and using no more than N function measurements. X0
% can be a scalar or a vector, FUN has to return a scalar value.
% 
% X_min = DONE(FUN,X0,N,LB,UB) minimizes FUN in the range
% lb<=x<=ub. The bounds LB and UB can be scalar or the same size as X0. When
% no bounds are given, the default bounds of -1 and 1 are used.
%
% X_min = DONE(FUN,X0,N,LB,UB,D) minimizes FUN using D random
% fourier expansions. Increasing the integer D can improve accuracy but also
% increases the running time. Default D=1000.
%
% X_min = DONE(FUN,X0,N,LB,UB,D,LAMBDA) adds a regularization parameter
% to the fit of the random Fourier expansion. Default LAMBDA=0.1.
%
% X_min = DONE(FUN,X0,N,LB,UB,D,LAMBDA,SIGMA) changes the variance of
% the frequencies of the random Fourier expansions. If FUN is very
% noisy, try decreasing SIGMA. Default SIGMA=1.
%
% X_min = DONE(FUN,X0,N,LB,UB,D,LAMBDA,SIGMA,EXPL) changes the
% exploration parameter. Increasing EXPL can be useful if the algorithm
% gets stuck in local minima. Default EXPL=0.1*sqrt(3)/d, where d is the
% length of X0.
%
% [X_min, MEASUREMENTS] = DONE(...) returns the X and Y measurements that were
% used in the algorithm. The Y measurements appear in the last row of
% MEASUREMENTS.
%
% [X_min, MEASUREMENTS, OPTS] = DONE(...) returns the optimal X and Y values that were
% found by the algorithm. The Y values appear in the last row of OPTS.
%
% [X_min, MEASUREMENTS, OPTS, MODEL] = DONE(...) returns the random Fourier expansion
% model.
%
% Current version (11/07/2016): 1.0
% Laurens Bliek & Hans Verstraete, 2016.

%% Initialization
setparameters(nargin);

next_X = X0(:); % X value used to take a measurement
MEASUREMENTS = zeros(MODEL.d+1,N); % X and Y values of measurements
OPTS = zeros(MODEL.d+1,N); % X and Y values of found minima
X_min = zeros(MODEL.d,1); % Current minimum

initializeModel();

%% DONE algorithm
for n = 1:N
    %% Store X (vector of length d) and Y (scalar) values in DATASET
    MEASUREMENTS(1:MODEL.d,n) = next_X;
    MEASUREMENTS(MODEL.d+1,n) = FUN(next_X);
    
    %% Update the RFE model using inverse QR
    updateModel(MEASUREMENTS(:,n));
    
    %% Exploration on the RFE model, initial guess for nonlinear optimization
    X0 = min(max(MODEL.LB,next_X + MODEL.EXPL*randn(MODEL.d,1)),MODEL.UB);

    %% Find the minimum of the RFE 
    options = optimset('algorithm','interior-point','display','off','MaxITer',10,'GradObj','on','Hessian','lbfgs');
    X_min = fmincon({MODEL.out,MODEL.deriv},X0, [], [],[],[],MODEL.LB,MODEL.UB,[],options);
    
    %% Store X (vector of length d) and Y (scalar) values in DATASET
    OPTS(1:MODEL.d,n) = X_min;
    OPTS(MODEL.d+1,n) = FUN(X_min);
    
    %% Exploration on FUN, choose new measurement point
    next_X = min(MODEL.UB,max(MODEL.LB,X_min + MODEL.EXPL*randn(MODEL.d,1))); 
end
    %% 
    function setparameters(n)
        %% Set default parameters
        MODEL.d = length(X0);	% X dimension
        if n < 9
            MODEL.EXPL = 0.1*sqrt(3)/MODEL.d; % Exploration parameter
            if n < 8
                MODEL.SIGMA = 1; % Standard deviation of random Fourier frequencies
                if n < 7
                    MODEL.LAMBDA = 0.1; % Regularization parameter
                    if n < 6
                        MODEL.D = 1000; % Number of basis functions of random Fourier expansion
                        if n < 4
                            MODEL.LB = -1*ones(MODEL.d,1); % Lower bound for the elements of x
                            MODEL.UB = 1*ones(MODEL.d,1); % Upper bound for the elements of x
                        end
                    end
                end      
            end  
        end

        %% Store parameters in RFE
        if n>=4
            MODEL.LB = LB;
            MODEL.UB = UB;
            if n >= 6
                MODEL.D = D;
                if n >= 7
                    MODEL.LAMBDA = LAMBDA;
                    if n >= 8
                        MODEL.SIGMA = SIGMA;
                        if n >= 9
                            MODEL.EXPL = EXPL;
                        end
                    end
                end
            end
        end

        %% Allow scalar values of LB and UB
        if length(MODEL.LB)==1 && MODEL.d>1
            MODEL.LB = MODEL.LB*ones(MODEL.d,1);
        end
        if length(MODEL.UB)==1 && MODEL.d>1
            MODEL.UB = MODEL.UB*ones(MODEL.d,1);
        end
    end
    %%
    function initializeModel()    
        %% Generate cosines with random frequencies and phases
        MODEL.OMEGA = MODEL.SIGMA*randn(MODEL.D,MODEL.d); % Frequencies
        MODEL.B = 2*pi*rand(MODEL.D,1); % Phases
        MODEL.Z = @(x) cos(MODEL.OMEGA*x+repmat(MODEL.B,1,size(x,2))); % Basis functions

        %% Initialize recursive least squares parameters
        MODEL.W = zeros(MODEL.D,1); % Weights of least squares solution
        MODEL.out = @(x2) MODEL.W'*MODEL.Z(x2); % Output of RFE
        MODEL.deriv = @(x) -MODEL.OMEGA'*diag(MODEL.W)*sin(MODEL.OMEGA*x+MODEL.B); % Derivative of RFE
        MODEL.P12 = 1/sqrt(MODEL.LAMBDA)*eye(MODEL.D); % Square root factor of P matrix
    end
    %%
    function updateModel(DATA)
        %% Read data
        U = MODEL.Z(DATA(1:MODEL.d))';
        Y = DATA(MODEL.d+1);
        
        %% Inverse QR update
        A =  [ 1,              U*MODEL.P12;   ...
               zeros(MODEL.D,1), MODEL.P12       ];
        [~,R ] = qr(A');
        R = R';
        gamma_12 = R(1,1);
        ggamma_12 = R(2:end,1);
        MODEL.P12 = R(2:end,2:end);
        MODEL.W = MODEL.W + ggamma_12/gamma_12*(Y-U*MODEL.W);
        MODEL.out = @(x2) MODEL.W'*MODEL.Z(x2);
        MODEL.deriv = @(x) -MODEL.OMEGA'*diag(MODEL.W)*sin(MODEL.OMEGA*x+MODEL.B);
    end
end