%% Minimize a noisy two-dimensional parabola with DONE

func = @(x) x'*x+0.03*randn; %parabola
x0 = [0.7; -0.9]; %initial guess
N = 20; %number of measurements

%Plot function
[X,Y] = meshgrid(-1:0.05:1);
Z = zeros(41,41);
for i=1:41
    for j=1:41
        Z(i,j) = func([X(i,j); Y(i,j)]);
    end
end
clf;
surf(X,Y,Z); 

Xmin_fmincon = fmincon(func, x0, [], []); %minimize with fmincon
Xmin_DONE = DONE(func, x0, N); %minimize with DONE

%Show results
hold on;
plot3(0,0,0,'o','MarkerSize',12);
plot3(Xmin_fmincon(1),Xmin_fmincon(2),func(Xmin_fmincon),'o','MarkerSize',11);
plot3(Xmin_DONE(1),Xmin_DONE(2),func(Xmin_DONE),'o','MarkerSize',11);
legend('Function','True minimum', 'fmincon minimum', 'DONE minimum');
hold off;