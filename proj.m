% خواندن و پردازش داده‌ها
load('Cleansing.mat');
cleans = Cleansing;



true_cleans = calculate_true_cleans(cleans);
true_cleans = repelem(true_cleans, 24);

days = (1:365)';
days = repelem(days, 24);

load('Radiation1400New.mat');
Radiation1400New = Radiation1400New(:);

load('Rain.mat');
Rain = Rain1400(:);
Rain = repelem(Rain, 24);

load('Temperature.mat');
Temperature = Temperature1400(:);
Temperature = repelem(Temperature, 3);

load('Power.mat');
Power = Power1400(:);

% اطمینان از ابعاد مناسب
true_cleans = reshape(true_cleans, [], 1);
days = reshape(days, [], 1);
Radiation1400New = reshape(Radiation1400New, [], 1);
Rain = reshape(Rain, [], 1);
Temperature = reshape(Temperature, [], 1);
Power = reshape(Power, [], 1);

% ترکیب داده‌ها
input_data = [true_cleans, days, Radiation1400New, Rain, Temperature];
output_data = Power;

% ذخیره ماتریس همبستگی
correlation_matrix = corr(input_data);
save('correlation_matrix.mat', 'correlation_matrix');

% تقسیم داده‌ها به آموزش و تست
cv = cvpartition(size(input_data,1), 'HoldOut', 0.2);
X_train = input_data(training(cv), :);
X_test = input_data(test(cv), :);
y_train = output_data(training(cv));
y_test = output_data(test(cv));

% مدل رگرسیون تصادفی
model = TreeBagger(100, X_train, y_train, 'Method', 'regression');

y_pred = predict(model, X_test);

% ارزیابی مدل
mse = mean((y_test - y_pred).^2);
rmse = sqrt(mse);
mae = mean(abs(y_test - y_pred));
r2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);

fprintf('Mean Squared Error: %f\n', mse);
fprintf('Root Mean Squared Error: %f\n', rmse);
fprintf('Mean Absolute Error: %f\n', mae);
fprintf('R² Score: %f\n', r2);

% رسم نمودارها
figure;
subplot(2,2,1);
scatter(y_test, y_pred);
xlabel('Actual Power'); ylabel('Predicted Power');
title('Actual vs Predicted Power');

subplot(2,2,2);
residuals = y_test - y_pred;
histogram(residuals);
xlabel('Residuals'); ylabel('Frequency');
title('Residuals Histogram');

subplot(2,2,3);
plot(y_test, 'b'); hold on;
plot(y_pred, 'r');
legend('Actual', 'Predicted');
title('Actual vs Predicted Power Over Samples');

subplot(2,2,4);
plot(residuals);
xlabel('Sample Index'); ylabel('Residuals');
title('Residuals Over Samples');

saveas(gcf, 'regression_plots.png');

function true_cleans = calculate_true_cleans(cleans)
    true_cleans = zeros(365, 1);
    ranges = [39, 79, 106, 121, 150, 162, 178, 193, 206, 365];
    for i = 1:365
        for r = ranges
            if i < r
                true_cleans(i) = r - i;
            end
        end
    end
end
