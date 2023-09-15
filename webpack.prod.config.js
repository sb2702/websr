const path = require('path');
const webpack = require('webpack');

const { CleanWebpackPlugin } = require('clean-webpack-plugin');


module.exports = {
    entry: [ "./src/main.ts"],
    mode: 'production',
    output: {
        library: "WebSR",
        libraryTarget: "umd",
        libraryExport: "default",
        path: path.resolve(__dirname, './dist'),
        chunkFilename: 'websr.[name].js',
        filename: "websr.js"
    },
    module: {

        rules: [
            {
                test: /\.ts?$/,
                use: {
                    loader: "ts-loader",
                    options: {
                        compilerOptions:{
                            "types": ["@webgpu/types"]
                        }
                    }
                },
                exclude: /node_modules/,

            },
        ],

    },

    plugins: [

        new CleanWebpackPlugin({
            cleanStaleWebpackAssets: false
        }),


    ],
    resolve: {
        extensions: [".ts", ".tsx", ".js"]
    },

    devServer: {
        static: {
            directory: path.join(__dirname, 'demo'),
        },
        compress: true,
        port: 8000,
    },

    mode: 'development'

};
